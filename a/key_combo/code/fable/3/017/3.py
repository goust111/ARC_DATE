from __future__ import annotations

from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_arc_utils():
    try:
        import arc_utils as mod  # type: ignore

        return mod
    except Exception:
        path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("arc_utils", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load arc_utils from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_utils = _load_arc_utils()
Color = _utils.Color
bbox_coords = _utils.bbox_coords
crop_bbox = _utils.crop_bbox
find_components = _utils.find_components
get_bg = _utils.get_bg
paste = _utils.paste


BG = Color.BLACK
MARKER = 4
SPRITE_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
TRANSFORM_ORDER = ("hflip", "vflip", "transpose", "rot180")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 600


def _apply_transform(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "hflip":
        return np.fliplr(arr).copy()
    if name == "vflip":
        return np.flipud(arr).copy()
    if name == "transpose":
        return arr.T.copy()
    if name == "rot180":
        return np.rot90(arr, 2).copy()
    raise ValueError(f"unknown transform: {name}")


def _component_crop(grid: np.ndarray, component: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = component["coords"]
    crop[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return crop


def _sprite_colors(crop: np.ndarray, bg: int) -> set[int]:
    return {int(x) for x in np.unique(crop) if int(x) != int(bg)}


def _derive_mapping(
    src: np.ndarray, dst: np.ndarray, bg: int, transform_name: str
) -> dict[int, int] | None:
    moved = _apply_transform(src, transform_name)
    if moved.shape != dst.shape:
        return None

    moved_mask = moved != int(bg)
    dst_mask = dst != int(bg)
    if not np.array_equal(moved_mask, dst_mask) or not bool(moved_mask.any()):
        return None

    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for r, c in np.argwhere(moved_mask):
        a = int(moved[int(r), int(c)])
        b = int(dst[int(r), int(c)])
        if a in mapping and mapping[a] != b:
            return None
        if b in reverse and reverse[b] != a:
            return None
        mapping[a] = b
        reverse[b] = a

    if not mapping or all(src_col == dst_col for src_col, dst_col in mapping.items()):
        return None
    return mapping


def _recolor(crop: np.ndarray, mapping: dict[int, int], bg: int) -> np.ndarray:
    out = crop.copy()
    for r, c in np.argwhere(out != int(bg)):
        color = int(out[int(r), int(c)])
        assert color in mapping, "marked sprite uses a color outside the key domain"
        out[int(r), int(c)] = int(mapping[color])
    return out


def _touches_marker(coords: np.ndarray, marker: tuple[int, int]) -> bool:
    mr, mc = marker
    for r, c in coords:
        if abs(int(r) - int(mr)) + abs(int(c) - int(mc)) == 1:
            return True
    return False


def _extract_marked_and_unmarked(grid: np.ndarray, bg: int) -> tuple[dict, list[dict]]:
    marker_cells = np.argwhere(grid == MARKER)
    assert marker_cells.shape == (1, 2), "expected exactly one marker pixel of color 4"
    marker = tuple(map(int, marker_cells[0]))

    without_marker = grid.copy()
    without_marker[marker] = int(bg)
    components = find_components(
        without_marker,
        bg=int(bg),
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) >= 3, "expected the marked sprite, a key pair, and optional distractors"

    sprites: list[dict] = []
    marked_indices: list[int] = []
    for idx, comp in enumerate(components):
        crop = _component_crop(without_marker, comp, bg)
        sprite = {
            "bbox": tuple(map(int, comp["bbox"])),
            "coords": comp["coords"],
            "crop": crop,
            "colors": _sprite_colors(crop, bg),
        }
        sprites.append(sprite)
        if _touches_marker(comp["coords"], marker):
            marked_indices.append(idx)

    assert len(marked_indices) == 1, "exactly one sprite must touch the marker"
    marked_idx = marked_indices[0]
    marked = sprites[marked_idx]
    unmarked = [sprite for i, sprite in enumerate(sprites) if i != marked_idx]
    return marked, unmarked


def _key_candidates(grid: np.ndarray) -> tuple[dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    marked, unmarked = _extract_marked_and_unmarked(g, bg)
    marked_colors = set(marked["colors"])

    candidates: list[dict] = []
    for src_i, src in enumerate(unmarked):
        for dst_i, dst in enumerate(unmarked):
            if src_i == dst_i:
                continue
            for transform_name in TRANSFORM_ORDER:
                mapping = _derive_mapping(src["crop"], dst["crop"], bg, transform_name)
                if mapping is None:
                    continue
                if marked_colors.issubset(set(mapping.keys())):
                    candidates.append(
                        {
                            "src": src_i,
                            "dst": dst_i,
                            "transform": transform_name,
                            "mapping": mapping,
                        }
                    )
    return marked, candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the paired-sprite geometric and color key to the marker-adjacent sprite.

    Input structure:
    - A dominant background surrounds separated multicolor sprites.
    - Exactly one sprite is 4-adjacent to a single marker pixel of color 4.
    - Among the unmarked sprites, exactly one ordered pair matches after one involutive
      transform: left-right reflection, top-bottom reflection, main-diagonal transpose,
      or 180-degree rotation.
    - That same ordered pair also gives a non-identity one-to-one color mapping, and the
      marked sprite uses only colors from the mapping domain.

    Rule:
    - Remove the marker and crop every sprite to its tight bounding box, preserving holes.
    - Find the unique unmarked ordered pair whose transformed source mask equals the target
      mask and whose colors define a consistent non-identity bijection.
    - Transform the marked sprite crop by that pair's transform, then recolor each foreground
      pixel with the pair's color bijection.

    Key insight:
    - The repeated pair is not the answer itself; it is a two-part key encoding both the
      reflection/rotation and the color renaming to apply to the marked sprite.
    """
    g = np.asarray(grid, dtype=int)
    marked, candidates = _key_candidates(g)
    assert len(candidates) == 1, "expected exactly one usable paired-sprite key"
    key = candidates[0]

    bg = int(get_bg(g))
    transformed = _apply_transform(marked["crop"], key["transform"])
    return _recolor(transformed, key["mapping"], bg)


# ---------------------------- generator ----------------------------


def _mask_connected(mask: np.ndarray) -> bool:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not pts:
        return False
    seen = {pts[0]}
    stack = [pts[0]]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == len(pts)


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected_mask(h: int, w: int, min_area: int, max_area: int) -> np.ndarray | None:
    max_area = min(int(max_area), int(h * w))
    min_area = max(1, min(int(min_area), max_area))
    for _ in range(200):
        target = random.randint(min_area, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _full_bbox(mask) and _mask_connected(mask):
            return mask
    return None


def _colorize_mask(mask: np.ndarray, colors: list[int], *, require_repeat: bool = False) -> np.ndarray | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < len(colors) or (require_repeat and len(coords) <= len(colors)):
        return None

    random.shuffle(coords)
    shuffled_colors = list(map(int, colors))
    random.shuffle(shuffled_colors)
    crop = np.full(mask.shape, BG, dtype=int)

    for cell, color in zip(coords, shuffled_colors):
        crop[cell] = int(color)

    rest = coords[len(shuffled_colors) :]
    if require_repeat and rest:
        crop[rest[0]] = int(shuffled_colors[0])
        rest = rest[1:]
    for cell in rest:
        crop[cell] = int(random.choice(shuffled_colors))
    return crop


def _mapping_signature(mapping: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(a), int(b)) for a, b in mapping.items()))


def _pair_transform_candidates(src: np.ndarray, dst: np.ndarray) -> list[tuple[str, dict[int, int]]]:
    found: list[tuple[str, dict[int, int]]] = []
    for name in TRANSFORM_ORDER:
        mapping = _derive_mapping(src, dst, BG, name)
        if mapping is not None:
            found.append((name, mapping))
    return found


def _make_key_pair(
    transform_name: str, mapping: dict[int, int], domain_colors: list[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(300):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        min_area = max(len(domain_colors) + 2, h + w - 2)
        mask = _grow_connected_mask(h, w, min_area, h * w - 2)
        if mask is None:
            continue
        src = _colorize_mask(mask, domain_colors, require_repeat=True)
        if src is None:
            continue

        moved = _apply_transform(src, transform_name)
        dst = _recolor(moved, mapping, BG)
        candidates = _pair_transform_candidates(src, dst)
        if len(candidates) != 1:
            continue
        name, found_mapping = candidates[0]
        if name == transform_name and _mapping_signature(found_mapping) == _mapping_signature(mapping):
            return src, dst
    return None


def _make_marked_sprite(transform_name: str, domain_colors: list[int]) -> np.ndarray | None:
    for _ in range(250):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        min_area = max(len(domain_colors), h + w - 2)
        mask = _grow_connected_mask(h, w, min_area, h * w - 1)
        if mask is None:
            continue
        transformed_mask = _apply_transform(mask.astype(int), transform_name).astype(bool)
        if transformed_mask.shape == mask.shape and np.array_equal(transformed_mask, mask):
            continue
        crop = _colorize_mask(mask, domain_colors)
        if crop is not None:
            return crop
    return None


def _make_bad_color_decoy(
    src: np.ndarray, transform_name: str, codomain_colors: list[int]
) -> np.ndarray | None:
    moved = _apply_transform(src, transform_name)
    decoy = np.full(moved.shape, BG, dtype=int)
    mask = moved != BG
    for r, c in np.argwhere(mask):
        decoy[int(r), int(c)] = int(random.choice(codomain_colors))

    for color in random.sample(list(_sprite_colors(moved, BG)), len(_sprite_colors(moved, BG))):
        cells = [tuple(map(int, rc)) for rc in np.argwhere(moved == int(color))]
        if len(cells) >= 2 and len(codomain_colors) >= 2:
            a, b = random.sample(cells, 2)
            x, y = random.sample(codomain_colors, 2)
            decoy[a] = int(x)
            decoy[b] = int(y)
            return decoy
    return None


def _make_random_distractor() -> np.ndarray | None:
    for _ in range(150):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        area_cap = max(1, h * w - 1)
        mask = _grow_connected_mask(h, w, min(4, area_cap), area_cap)
        if mask is None:
            continue
        n_colors = random.randint(1, min(3, int(mask.sum()), len(SPRITE_COLORS)))
        colors = random.sample(SPRITE_COLORS, n_colors)
        crop = _colorize_mask(mask, colors)
        if crop is not None:
            return crop
    return None


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _paste_crop(grid: np.ndarray, crop: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, crop, int(r0), int(c0), transparent=BG)


def _candidate_locations(
    shape: tuple[int, int], crop: np.ndarray, reserved: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]]:
    H, W = shape
    h, w = crop.shape
    locations: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_bbox_separated(box, other) for other in reserved):
                locations.append((r0, c0))
    random.shuffle(locations)
    return locations


def _place_regular_sprite(
    grid: np.ndarray, crop: np.ndarray, reserved: list[tuple[int, int, int, int]]
) -> bool:
    for r0, c0 in _candidate_locations(grid.shape, crop, reserved):
        _paste_crop(grid, crop, r0, c0)
        h, w = crop.shape
        reserved.append((r0, c0, r0 + h - 1, c0 + w - 1))
        return True
    return False


def _marker_offsets(crop: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = crop.shape
    options: list[tuple[int, int, int, int]] = []
    for r, c in np.argwhere(crop != BG):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            mr, mc = rr + dr, cc + dc
            if mr < 0 or mr >= h or mc < 0 or mc >= w:
                options.append((rr, cc, mr, mc))
    random.shuffle(options)
    return options


def _place_marked_sprite(
    grid: np.ndarray, crop: np.ndarray, reserved: list[tuple[int, int, int, int]]
) -> bool:
    H, W = grid.shape
    h, w = crop.shape
    top_lefts = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(top_lefts)
    offsets = _marker_offsets(crop)
    if not offsets:
        return False

    for r0, c0 in top_lefts:
        sprite_box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if not all(_bbox_separated(sprite_box, other) for other in reserved):
            continue
        for _sr, _sc, mr, mc in offsets:
            ar, ac = r0 + mr, c0 + mc
            if not (0 <= ar < H and 0 <= ac < W):
                continue
            marker_box = (ar, ac, ar, ac)
            if int(grid[ar, ac]) != BG:
                continue
            if not all(_bbox_separated(marker_box, other) for other in reserved):
                continue
            _paste_crop(grid, crop, r0, c0)
            grid[ar, ac] = MARKER
            reserved.append(sprite_box)
            reserved.append(marker_box)
            return True
    return False


def _expected_from_key(marked: np.ndarray, transform_name: str, mapping: dict[int, int]) -> np.ndarray:
    return _recolor(_apply_transform(marked, transform_name), mapping, BG)


def _make_attempt() -> np.ndarray | None:
    transform_name = random.choice(TRANSFORM_ORDER)
    n_colors = random.choice([2, 2, 3, 3, 4])
    palette = random.sample(SPRITE_COLORS, 2 * n_colors)
    domain_colors = palette[:n_colors]
    codomain_colors = palette[n_colors:]
    shuffled_codomain = codomain_colors[:]
    random.shuffle(shuffled_codomain)
    mapping = {int(a): int(b) for a, b in zip(domain_colors, shuffled_codomain)}

    key_pair = _make_key_pair(transform_name, mapping, domain_colors)
    if key_pair is None:
        return None
    key_src, key_dst = key_pair

    marked = _make_marked_sprite(transform_name, domain_colors)
    if marked is None:
        return None
    expected = _expected_from_key(marked, transform_name, mapping)

    distractors: list[np.ndarray] = []
    if random.random() < 0.8:
        decoy = _make_bad_color_decoy(key_src, transform_name, codomain_colors)
        if decoy is not None:
            distractors.append(decoy)
    for _ in range(random.randint(1, 4)):
        distractor = _make_random_distractor()
        if distractor is None:
            return None
        distractors.append(distractor)

    H = random.randint(15, 30)
    W = random.randint(15, 30)
    grid = np.full((H, W), BG, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    if not _place_marked_sprite(grid, marked, reserved):
        return None

    remaining = [key_src, key_dst] + distractors
    random.shuffle(remaining)
    remaining.sort(key=lambda crop: int(crop.size), reverse=True)
    for crop in remaining:
        if not _place_regular_sprite(grid, crop, reserved):
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Check that the foreground interpretation is semantic, not merely self-consistent.
    marked_sprite, candidates = _key_candidates(grid)
    if len(candidates) != 1:
        return None
    key = candidates[0]
    if key["transform"] != transform_name:
        return None
    if _mapping_signature(key["mapping"]) != _mapping_signature(mapping):
        return None
    if not np.array_equal(_expected_from_key(marked_sprite["crop"], transform_name, mapping), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-sprite key input")

from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 700
D4_NAMES = ("r0", "r0m", "r1", "r1m", "r2", "r2m", "r3", "r3m")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors(r, c, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform(a: np.ndarray, name: str) -> np.ndarray:
    k = int(name[1])
    out = np.rot90(a, k)
    if name.endswith("m"):
        out = np.fliplr(out)
    return out.copy()


def _inverse_name(forward: str, shape: tuple[int, int]) -> str:
    ids = np.arange(shape[0] * shape[1]).reshape(shape)
    moved = _transform(ids, forward)
    matches = [
        name
        for name in D4_NAMES
        if _transform(moved, name).shape == ids.shape and np.array_equal(_transform(moved, name), ids)
    ]
    assert len(matches) == 1, "dihedral inverse must be unique"
    return matches[0]


def _solid_rectangle_colors(g: np.ndarray, bg: int) -> set[int]:
    out: set[int] = set()
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        mask = g == int(color)
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if int(mask.sum()) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue
        if r1 > r0 and c1 > c0:
            out.add(int(color))
    return out


def _extract_fragments(g: np.ndarray, bg: int) -> list[dict]:
    occluder_colors = _solid_rectangle_colors(g, bg)
    assert len(occluder_colors) == 3, "expected exactly three solid rectangular occluder colors"

    fragments: list[dict] = []
    for coords in _components(g != bg, connectivity=8):
        colors = {int(g[r, c]) for r, c in coords}
        occ_here = colors & occluder_colors
        assert len(occ_here) == 1, "each transformed copy must contain one occluder"
        occ_color = next(iter(occ_here))
        box = _bbox(coords)
        patch = _crop(g, box)
        occ = patch == occ_color

        occ_coords = np.argwhere(occ)
        assert occ_coords.size > 0, "occluder rectangle must be visible"
        or0, oc0, or1, oc1 = _bbox(occ_coords)
        assert int(occ.sum()) == (or1 - or0 + 1) * (oc1 - oc0 + 1), "occluder must be a solid rectangle"
        assert np.any((patch != bg) & (~occ)), "each copy must have visible sprite cells"

        fragments.append({"bbox": box, "patch": patch, "occ": occ, "occ_color": occ_color})

    assert len(fragments) == 3, "expected three separated transformed copy fragments"
    fragments.sort(key=lambda item: item["bbox"])
    return fragments


def _alignment_candidates(src: dict, dst: dict) -> list[str]:
    out: list[str] = []
    src_patch = src["patch"]
    src_occ = src["occ"]
    dst_patch = dst["patch"]
    dst_occ = dst["occ"]

    for name in D4_NAMES:
        moved = _transform(src_patch, name)
        moved_occ = _transform(src_occ, name)
        if moved.shape != dst_patch.shape:
            continue
        known = (~moved_occ) & (~dst_occ)
        min_known = max(6, min(int(moved.size // 4), 12))
        if int(known.sum()) < min_known:
            continue
        if np.array_equal(moved[known], dst_patch[known]):
            out.append(name)
    return out


def _complete_in_canonical(fragments: list[dict], names: list[str], bg: int) -> np.ndarray:
    shape = fragments[0]["patch"].shape
    aligned: list[tuple[np.ndarray, np.ndarray]] = []
    for frag, name in zip(fragments, names):
        inv = _inverse_name(name, shape)
        patch = _transform(frag["patch"], inv)
        occ = _transform(frag["occ"], inv)
        assert patch.shape == shape and occ.shape == shape, "all copies must align to one template"
        aligned.append((patch, occ))

    full = np.full(shape, bg, dtype=int)
    visible_count = np.zeros(shape, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            vals = [int(patch[r, c]) for patch, occ in aligned if not bool(occ[r, c])]
            if vals:
                assert len(set(vals)) == 1, "visible transformed copies must agree"
                full[r, c] = vals[0]
                visible_count[r, c] = len(vals)

    assert np.any(full != bg), "completed sprite must contain foreground colors"
    assert int((visible_count == 1).sum()) > 0, "some cells must require the third copy as sole donor"
    return full


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore three occluded transformed copies of the same multicolor sprite by cross-orientation union.

    Input structure:
    - The dominant color is the background.
    - Three separated multicolor sprite copies are present; their orientations are distinct
      dihedral transforms of one asymmetric template.
    - Each copy contains one solid rectangular occluder, and the three occluder colors are
      distinct from the sprite colors.

    Rule:
    - Use the visible non-occluded cells to find the unique dihedral correspondence from one
      copy's local coordinate system to each other copy.
    - In that common coordinate system, collect every visible cell value from the three copies;
      any cell still unseen is treated as background, so hidden non-sprite holes are erased.
    - Replace every occluder rectangle cell in each original copy by the corresponding restored
      sprite or background value, leaving all other cells fixed.

    Key insight:
    - The missing colors are not located by translation or by one chosen donor copy; the same
      template cell must be followed through three different mirrored or rotated coordinate
      systems, and some hidden cells have only the third orientation as a visible donor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"
    assert int(g.min()) >= 0 and int(g.max()) <= 9, "expected ARC palette colors"

    bg = _get_bg(g)
    fragments = _extract_fragments(g, bg)

    names = ["r0"]
    for frag in fragments[1:]:
        candidates = _alignment_candidates(fragments[0], frag)
        assert len(candidates) == 1, "each copy orientation must be uniquely determined"
        names.append(candidates[0])
    assert len(set(names)) == 3, "the three observed copy transforms must be distinct"

    canonical = _complete_in_canonical(fragments, names, bg)

    out = g.copy()
    for frag, name in zip(fragments, names):
        expected = _transform(canonical, name)
        patch = frag["patch"]
        occ = frag["occ"]
        assert expected.shape == patch.shape, "restored copy must match the observed bbox"
        known = ~occ
        assert np.array_equal(expected[known], patch[known]), "restored copy must preserve visible cells"

        r0, c0, r1, c1 = frag["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[occ] = expected[occ]
    return out


# --------------- generator ---------------


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=4)
    return len(comps) == 1


def _has_symmetry(sprite: np.ndarray) -> bool:
    for name in D4_NAMES:
        if name == "r0":
            continue
        moved = _transform(sprite, name)
        if moved.shape == sprite.shape and np.array_equal(moved, sprite):
            return True
    return False


def _rect_mask(h: int, w: int, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _random_box(h: int, w: int) -> tuple[int, int, int, int]:
    rh = random.randint(2, max(2, min(h - 2, h // 2 + 1)))
    rw = random.randint(2, max(2, min(w - 2, w // 2 + 1)))
    r0 = random.randint(1, h - rh - 1)
    c0 = random.randint(1, w - rw - 1)
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _choose_cover_masks(h: int, w: int) -> list[np.ndarray] | None:
    for _ in range(1200):
        masks = [_rect_mask(h, w, _random_box(h, w)) for _ in range(3)]
        triple = masks[0] & masks[1] & masks[2]
        if not triple.any():
            continue
        if any(int(mask.sum()) > (h * w) // 3 for mask in masks):
            continue
        ok = True
        for i in range(3):
            for j in range(i + 1, 3):
                k = 3 - i - j
                if not (masks[i] & masks[j] & ~masks[k]).any():
                    ok = False
        if ok:
            return masks
    return None


def _build_template(h: int, w: int, masks: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    triple = masks[0] & masks[1] & masks[2]
    pair_only = []
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            pair_only.append(masks[i] & masks[j] & ~masks[k])

    forced_fg: set[tuple[int, int]] = set()
    for zone in pair_only:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(zone)]
        if not coords:
            return None
        forced_fg.add(random.choice(coords))

    border_choices = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    forced_fg.update(border_choices)

    for _ in range(300):
        mask = np.random.random((h, w)) < random.uniform(0.48, 0.66)
        mask[triple] = False
        for r, c in forced_fg:
            mask[r, c] = True
        if int(mask.sum()) < max(12, (h * w) // 3):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _is_connected(mask):
            continue
        if np.any(mask & triple):
            continue
        if any(not np.any(mask & zone) for zone in pair_only):
            continue

        sprite = np.full((h, w), BG, dtype=int)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            sprite[r, c] = colors[idx % len(colors)]
        for r, c in coords:
            if random.random() < 0.45:
                sprite[r, c] = random.choice(colors)
        if any(not np.any(sprite == color) for color in colors):
            continue
        if _has_symmetry(sprite):
            continue
        return sprite
    return None


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _pad_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _place_shapes(shapes: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}
    blocked: list[tuple[int, int, int, int]] = []
    for idx in order:
        ph, pw = shapes[idx]
        choices: list[tuple[int, int]] = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                padded = _pad_box(box, 2, (h, w))
                if any(_overlap(padded, other) for other in blocked):
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        loc = random.choice(choices)
        placed[idx] = loc
        r0, c0 = loc
        blocked.append(_pad_box((r0, c0, r0 + ph - 1, c0 + pw - 1), 2, (h, w)))
    return [placed[i] for i in range(len(shapes))]


def _render_scene(
    sprite: np.ndarray,
    masks: list[np.ndarray],
    names: list[str],
    occ_colors: list[int],
    locs: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for mask, name, occ_color, (r0, c0) in zip(masks, names, occ_colors, locs):
        full = _transform(sprite, name)
        occ = _transform(mask, name)
        patch_in = full.copy()
        patch_in[occ] = int(occ_color)
        ph, pw = full.shape
        region_in = inp[r0 : r0 + ph, c0 : c0 + pw]
        region_out = out[r0 : r0 + ph, c0 : c0 + pw]
        region_in[patch_in != BG] = patch_in[patch_in != BG]
        region_out[full != BG] = full[full != BG]
    return inp, out


def _component_bboxes_match(inp: np.ndarray, sprite: np.ndarray, names: list[str]) -> bool:
    comps = _components(inp != BG, connectivity=8)
    if len(comps) != 3:
        return False
    expected_shapes = sorted(_transform(sprite, name).shape for name in names)
    actual_shapes = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        actual_shapes.append((r1 - r0 + 1, c1 - c0 + 1))
    return sorted(actual_shapes) == expected_shapes


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    sprite: np.ndarray,
    masks: list[np.ndarray],
    names: list[str],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if len(set(names)) != 3:
        return False
    fg = sprite != BG
    triple = masks[0] & masks[1] & masks[2]
    if np.any(fg & triple):
        return False
    for i in range(3):
        if not np.any(masks[i] & fg):
            return False
        visible_fg = fg & ~masks[i]
        if int(visible_fg.sum()) < 8:
            return False
    for i in range(3):
        for j in range(i + 1, 3):
            k = 3 - i - j
            if not np.any(fg & masks[i] & masks[j] & ~masks[k]):
                return False
    if not _component_bboxes_match(inp, sprite, names):
        return False

    counts = Counter(map(int, inp.ravel()))
    if counts[BG] <= max(v for c, v in counts.items() if c != BG):
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


_GEN_OFFSET = random.randrange(len(D4_NAMES))
_GEN_CALLS = 0


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(6, 8)
    w = random.randint(6, 9)
    if h == w and random.random() < 0.65:
        w = min(9, w + 1)

    masks = _choose_cover_masks(h, w)
    if masks is None:
        return None

    palette = random.sample(range(1, 10), 6)
    sprite_colors = palette[:3]
    occ_colors = palette[3:]
    sprite = _build_template(h, w, masks, sprite_colors)
    if sprite is None:
        return None

    global _GEN_CALLS
    start = (_GEN_CALLS + _GEN_OFFSET) % len(D4_NAMES)
    _GEN_CALLS += 1
    base_order = list(D4_NAMES[start:] + D4_NAMES[:start])
    if random.random() < 0.5:
        chosen = random.sample(D4_NAMES, 3)
    else:
        chosen = [base_order[0], base_order[3], base_order[5]]
    random.shuffle(chosen)

    shapes = [_transform(sprite, name).shape for name in chosen]
    grid_h = random.randint(max(20, max(s[0] for s in shapes) + 12), 30)
    grid_w = random.randint(max(20, max(s[1] for s in shapes) + 12), 30)
    locs = _place_shapes(shapes, grid_h, grid_w)
    if locs is None:
        return None

    color_perm = list(occ_colors)
    random.shuffle(color_perm)
    inp, expected = _render_scene(sprite, masks, chosen, color_perm, locs, (grid_h, grid_w))
    if not _semantic_check(inp, expected, sprite, masks, chosen):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")

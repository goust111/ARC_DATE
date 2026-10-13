from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _local_mask(shape: tuple[int, int], coords: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, _r1, _c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == bg:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells: list[Coord] = []
            colors: set[int] = set()
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                colors.add(int(grid[r, c]))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                    "colors": colors,
                }
            )
    return comps


def _connected_bool(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _scale_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask, np.ones((factor, factor), dtype=bool))


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _candidate_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    return _local_mask((r1 - r0 + 1, c1 - c0 + 1), comp["coords"], comp["bbox"])


def _match_component(template: np.ndarray, bg: int, comp: dict) -> list[tuple[int, int]]:
    cand_mask = _candidate_mask(comp)
    ch, cw = cand_mask.shape
    matches: list[tuple[int, int]] = []
    for k in range(4):
        rotated = np.rot90(template, k=k)
        rmask = rotated != bg
        rh, rw = rmask.shape
        if ch % rh != 0 or cw % rw != 0:
            continue
        scale_h = ch // rh
        scale_w = cw // rw
        if scale_h != scale_w or scale_h < 2:
            continue
        if np.array_equal(_scale_bool(rmask, scale_h), cand_mask):
            matches.append((k, scale_h))
    return matches


def _parse_scene(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict], list[tuple[int, int, int]]]:
    bg = _get_bg(grid)
    comps = _find_foreground_components(grid, bg)
    template_comps = [comp for comp in comps if len(comp["colors"]) >= 2]
    assert len(template_comps) == 1, "expected exactly one multicolor template"
    template_comp = template_comps[0]
    template = _crop_bbox(grid, template_comp["bbox"])

    candidates = [comp for comp in comps if comp is not template_comp]
    assert len(candidates) >= 2, "expected several candidate silhouettes"
    for comp in candidates:
        assert len(comp["colors"]) == 1, "candidate silhouettes must be monochrome"
        assert comp["area"] > template_comp["area"], "candidate silhouettes must be larger than the template"
    silhouette_colors = [next(iter(comp["colors"])) for comp in candidates]
    assert len(set(silhouette_colors)) == len(silhouette_colors), "silhouette colors must be distinct"

    matches: list[tuple[int, int, int]] = []
    for idx, comp in enumerate(candidates):
        for k, scale in _match_component(template, bg, comp):
            matches.append((idx, k, scale))
    return bg, template, candidates, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the only silhouette that is a rotated integer-scale copy of the small template.

    Input structure:
    - A background canvas contains one small connected multicolor template and several larger monochrome silhouettes.
    - Each silhouette has its own color; exactly one silhouette is the template foreground mask after a 90-degree rotation and square block scale.
    - The other silhouettes are reflected copies or one-to-two-block local edits of that shape.

    Rule:
    - Find the unique candidate whose mask equals a rotated, integer block-scaled version of the template foreground mask.
    - Rotate the full template by that same angle, scale every template cell into square blocks, and repaint the matching silhouette with those colors.
    - Erase every nonmatching silhouette to the background while leaving the original template and all background cells unchanged.

    Key insight:
    - The silhouette must belong to the rotation class of the template mask; mirror images and near-matches are decoys even when their size and outline look plausible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, template, candidates, matches = _parse_scene(g)
    assert len(matches) == 1, "expected exactly one rotation-scale silhouette match"

    true_idx, rotation, scale = matches[0]
    out = g.copy()
    for comp in candidates:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    true_comp = candidates[true_idx]
    r0, c0, _r1, _c1 = true_comp["bbox"]
    colored = _scale_sprite(np.rot90(template, k=rotation), scale)
    region = out[r0 : r0 + colored.shape[0], c0 : c0 + colored.shape[1]]
    fg = colored != bg
    region[fg] = colored[fg]
    return out


# --------------- generator ---------------

_BG = 0
_ALL_FG = list(range(1, 10))
_TEMPLATE_SIZES = [(3, 4), (4, 3), (4, 4), (3, 5), (5, 3)]
_MAX_ATTEMPTS = 500
_PLACE_ATTEMPTS = 300
_GEN_CALL_COUNT = 0


def _same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k=k) for k in range(4)]


def _reflections(mask: np.ndarray) -> list[np.ndarray]:
    return [np.fliplr(np.rot90(mask, k=k)) for k in range(4)]


def _equals_any(mask: np.ndarray, masks: list[np.ndarray]) -> bool:
    return any(_same_shape(mask, other) for other in masks)


def _valid_template_mask(mask: np.ndarray) -> bool:
    if not _connected_bool(mask) or not _full_bbox(mask):
        return False
    area = int(mask.sum())
    if area < 5 or area > mask.size - 2:
        return False
    rots = _rotations(mask)
    for i in range(4):
        for j in range(i + 1, 4):
            if _same_shape(rots[i], rots[j]):
                return False
    for refl in _reflections(mask):
        if _equals_any(refl, rots):
            return False
    return True


def _random_connected_mask(h: int, w: int) -> np.ndarray | None:
    target = random.randint(max(5, (h * w) // 3), max(6, (h * w * 2) // 3))
    mask = np.zeros((h, w), dtype=bool)
    r = random.randrange(h)
    c = random.randrange(w)
    mask[r, c] = True
    frontier = {(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w}
    while int(mask.sum()) < target and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    if _valid_template_mask(mask):
        return mask
    return None


def _make_template_mask() -> np.ndarray | None:
    sizes = _TEMPLATE_SIZES[:]
    random.shuffle(sizes)
    for h, w in sizes:
        for _ in range(80):
            mask = _random_connected_mask(h, w)
            if mask is not None:
                return mask
    return None


def _make_template(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assignments = colors[:] + [random.choice(colors) for _ in range(len(coords) - len(colors))]
    random.shuffle(assignments)
    template = np.full(mask.shape, _BG, dtype=int)
    for (r, c), color in zip(coords, assignments):
        template[r, c] = int(color)
    return template


def _try_modified_mask(base: np.ndarray, template_mask: np.ndarray) -> np.ndarray | None:
    rotation_class = _rotations(template_mask)
    h, w = base.shape
    for _ in range(120):
        mask = base.copy()
        changes = random.choice([1, 2])
        changed = False
        for _change in range(changes):
            options: list[np.ndarray] = []
            for r, c in np.argwhere(mask):
                trial = mask.copy()
                trial[int(r), int(c)] = False
                if _connected_bool(trial) and _full_bbox(trial) and int(trial.sum()) >= 5:
                    options.append(trial)
            for r, c in np.argwhere(~mask):
                rr = int(r)
                cc = int(c)
                if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(rr, cc)):
                    trial = mask.copy()
                    trial[rr, cc] = True
                    if _connected_bool(trial) and _full_bbox(trial) and int(trial.sum()) <= mask.size - 1:
                        options.append(trial)
            if not options:
                break
            mask = random.choice(options)
            changed = True
        if changed and not _equals_any(mask, rotation_class) and _connected_bool(mask) and _full_bbox(mask):
            return mask
    return None


def _shape_matches_template(template: np.ndarray, candidate_mask: np.ndarray) -> list[tuple[int, int]]:
    comp = {
        "coords": np.argwhere(candidate_mask),
        "bbox": (0, 0, candidate_mask.shape[0] - 1, candidate_mask.shape[1] - 1),
        "colors": {1},
        "area": int(candidate_mask.sum()),
    }
    return _match_component(template, _BG, comp)


def _make_silhouette_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, _BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _build_candidates(template: np.ndarray, template_mask: np.ndarray, scale: int, rotation: int, count: int):
    true_base = np.rot90(template_mask, k=rotation)
    silhouettes: list[dict] = [
        {
            "kind": "true",
            "base": true_base,
            "scale": scale,
            "rotation": rotation,
        }
    ]

    mirror_base = np.fliplr(true_base)
    if _shape_matches_template(template, _scale_bool(mirror_base, scale)):
        return None
    silhouettes.append({"kind": "mirror", "base": mirror_base, "scale": scale, "rotation": None})

    local_base = _try_modified_mask(true_base, template_mask)
    if local_base is None:
        return None
    if _shape_matches_template(template, _scale_bool(local_base, scale)):
        return None
    silhouettes.append({"kind": "modified", "base": local_base, "scale": scale, "rotation": None})

    while len(silhouettes) < count:
        if random.random() < 0.55:
            k = random.randrange(4)
            base = np.fliplr(np.rot90(template_mask, k=k))
            kind = "mirror"
        else:
            k = random.randrange(4)
            base0 = np.rot90(template_mask, k=k)
            base = _try_modified_mask(base0, template_mask)
            kind = "modified"
            if base is None:
                continue
        cand_scale = random.choice([scale, 2, 3])
        scaled = _scale_bool(base, cand_scale)
        if scaled.shape[0] > 18 or scaled.shape[1] > 18:
            continue
        if _shape_matches_template(template, scaled):
            continue
        silhouettes.append({"kind": kind, "base": base, "scale": cand_scale, "rotation": None})
    return silhouettes


def _place_rectangles(shape: tuple[int, int], rects: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    h, w = shape
    order = sorted(range(len(rects)), key=lambda i: rects[i][0] * rects[i][1], reverse=True)
    occ = np.zeros(shape, dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(rects)
    pad = 1
    for idx in order:
        rh, rw = rects[idx]
        possible: list[tuple[int, int]] = []
        for r in range(1, h - rh):
            for c in range(1, w - rw):
                r0 = max(0, r - pad)
                c0 = max(0, c - pad)
                r1 = min(h, r + rh + pad)
                c1 = min(w, c + rw + pad)
                if not occ[r0:r1, c0:c1].any():
                    possible.append((r, c))
        if not possible:
            return None
        pos = random.choice(possible)
        positions[idx] = pos
        r, c = pos
        occ[max(0, r - pad) : min(h, r + rh + pad), max(0, c - pad) : min(w, c + rw + pad)] = True
    return [pos for pos in positions if pos is not None]


def _render_scene(
    template: np.ndarray,
    silhouettes: list[dict],
    template_colors: list[int],
    silhouette_colors: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    patches: list[np.ndarray] = [template]
    true_output_patch = None
    for item, color in zip(silhouettes, silhouette_colors):
        mask = _scale_bool(item["base"], int(item["scale"]))
        patches.append(_make_silhouette_patch(mask, color))
        if item["kind"] == "true":
            true_output_patch = _scale_sprite(np.rot90(template, k=int(item["rotation"])), int(item["scale"]))
    assert true_output_patch is not None

    rects = [patch.shape for patch in patches]
    for _ in range(_PLACE_ATTEMPTS):
        height = random.randint(24, 30)
        width = random.randint(24, 30)
        if any(rh + 2 > height or rw + 2 > width for rh, rw in rects):
            continue
        positions = _place_rectangles((height, width), rects)
        if positions is None:
            continue

        inp = np.full((height, width), _BG, dtype=int)
        out = np.full((height, width), _BG, dtype=int)
        _paste(inp, template, positions[0][0], positions[0][1], transparent=_BG)
        _paste(out, template, positions[0][0], positions[0][1], transparent=_BG)

        for i, patch in enumerate(patches[1:], start=1):
            _paste(inp, patch, positions[i][0], positions[i][1], transparent=_BG)
        true_index = next(i for i, item in enumerate(silhouettes) if item["kind"] == "true")
        true_pos = positions[true_index + 1]
        _paste(out, true_output_patch, true_pos[0], true_pos[1], transparent=_BG)

        used_template_colors = set(map(int, template[template != _BG]))
        if used_template_colors != set(template_colors):
            return None
        if any(color in used_template_colors for color in silhouette_colors):
            return None
        return inp, out
    return None


def _gen_attempt(force_rotation: int | None = None, force_scale: int | None = None) -> np.ndarray | None:
    template_mask = _make_template_mask()
    if template_mask is None:
        return None

    candidate_count = random.choice([3, 3, 4])
    template_color_count = random.choice([3, 4])
    if candidate_count + template_color_count > len(_ALL_FG):
        template_color_count = len(_ALL_FG) - candidate_count
    palette = random.sample(_ALL_FG, template_color_count + candidate_count)
    template_colors = palette[:template_color_count]
    silhouette_colors = palette[template_color_count:]
    template = _make_template(template_mask, template_colors)

    rotation = random.randrange(4) if force_rotation is None else int(force_rotation) % 4
    scale_options = [2, 3]
    scale = random.choice(scale_options) if force_scale is None else int(force_scale)
    if scale not in scale_options:
        scale = 2
    if max(np.rot90(template_mask, k=rotation).shape) * scale > 18:
        scale = 2

    silhouettes = _build_candidates(template, template_mask, scale, rotation, candidate_count)
    if silhouettes is None:
        return None
    random.shuffle(silhouettes)
    if not any(item["kind"] == "mirror" and int(item["scale"]) == scale for item in silhouettes):
        return None
    if not any(item["kind"] == "modified" and int(item["scale"]) == scale for item in silhouettes):
        return None

    rendered = _render_scene(template, silhouettes, template_colors, silhouette_colors)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        bg, _template, candidates, matches = _parse_scene(inp)
        if bg != _BG or len(matches) != 1 or len(candidates) != candidate_count:
            return None
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    forced_rotation = call % 4
    forced_scale = 2 if (call // 4) % 2 == 0 else 3
    for attempt in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(
            force_rotation=forced_rotation if attempt < _MAX_ATTEMPTS // 2 else None,
            force_scale=forced_scale if attempt < _MAX_ATTEMPTS // 2 else None,
        )
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid true-body silhouette input")

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import crop_bbox, find_components, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(cells, dtype=int))
        return comps

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
        **_: object,
    ) -> list[dict]:
        assert connectivity == 4
        assert monochromatic
        assert return_format == "dict"
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        usable = sorted(set(map(int, grid.flatten())) - {bg}) if colors is None else [int(c) for c in colors]
        out: list[dict] = []
        for color in usable:
            for coords in _mask_components(grid == color):
                r0, c0 = coords.min(axis=0)
                r1, c1 = coords.max(axis=0)
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(coords.shape[0]),
                    }
                )
        return out


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _transform_mask(mask: np.ndarray, k: int, mirrored: bool) -> np.ndarray:
    arr = np.rot90(mask, int(k))
    if mirrored:
        arr = np.fliplr(arr)
    return np.asarray(arr, dtype=bool).copy()


def _transform_point(r: int, c: int, shape: tuple[int, int], k: int, mirrored: bool) -> tuple[int, int]:
    h, w = map(int, shape)
    rr = int(r)
    cc = int(c)
    if int(k) == 0:
        out_r, out_c = rr, cc
        out_h, out_w = h, w
    elif int(k) == 1:
        out_r, out_c = w - 1 - cc, rr
        out_h, out_w = w, h
    elif int(k) == 2:
        out_r, out_c = h - 1 - rr, w - 1 - cc
        out_h, out_w = h, w
    elif int(k) == 3:
        out_r, out_c = cc, h - 1 - rr
        out_h, out_w = w, h
    else:
        raise ValueError("rotation index must be 0..3")
    if mirrored:
        out_c = out_w - 1 - out_c
    return int(out_r), int(out_c)


def _dihedral_variants(mask: np.ndarray) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        for mirrored in (False, True):
            cur = _transform_mask(mask, k, mirrored)
            key = _shape_key(cur)
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "k": int(k),
                    "mirrored": bool(mirrored),
                    "mask": cur,
                    "key": key,
                    "name": f"rot{k}{'_flip' if mirrored else ''}",
                }
            )
    return variants


def _component_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _shadow_neighbors(grid: np.ndarray, comp: dict, shadow_color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    found: set[tuple[int, int]] = set()
    for rr, cc in np.asarray(comp["coords"], dtype=int):
        for nr, nc in _neighbors8(int(rr), int(cc)):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(shadow_color):
                found.add((int(nr), int(nc)))
    return sorted(found)


def _match_variant(source_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, object] | None:
    matches = [
        variant
        for variant in _dihedral_variants(source_mask)
        if variant["mask"].shape == target_mask.shape and np.array_equal(variant["mask"], target_mask)
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _parse_scene(
    grid: np.ndarray, bg: int, body_color: int, shadow_color: int
) -> dict[str, object] | None:
    comps = find_components(
        grid,
        bg=bg,
        colors=[body_color],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if len(comps) < 3:
        return None

    source_candidates: list[tuple[dict, list[tuple[int, int]]]] = []
    for comp in comps:
        shadows = _shadow_neighbors(grid, comp, shadow_color)
        if shadows:
            source_candidates.append((comp, shadows))
    if len(source_candidates) != 1:
        return None

    source_comp, source_shadows_abs = source_candidates[0]
    source_mask = _component_mask(grid, source_comp)
    if len(_dihedral_variants(source_mask)) != 8:
        return None

    sr0, sc0, _sr1, _sc1 = map(int, source_comp["bbox"])
    source_shadow_local = [(int(r - sr0), int(c - sc0)) for r, c in source_shadows_abs]
    if len(source_shadow_local) < 2:
        return None

    matched: list[dict[str, object]] = []
    distractors: list[dict] = []
    for comp in comps:
        mask = _component_mask(grid, comp)
        match = _match_variant(source_mask, mask)
        if match is None:
            distractors.append(comp)
            continue
        matched.append({"comp": comp, "variant": match, "mask": mask})

    if len(matched) < 3:
        return None
    if not distractors:
        return None

    variant_names = [str(item["variant"]["name"]) for item in matched if item["comp"] is not source_comp]
    has_quarter_turn = any(name.startswith("rot1") or name.startswith("rot3") for name in variant_names)
    has_mirror = any("_flip" in name for name in variant_names)
    if not has_quarter_turn or not has_mirror:
        return None

    return {
        "body_color": int(body_color),
        "shadow_color": int(shadow_color),
        "components": comps,
        "source_comp": source_comp,
        "source_mask": source_mask,
        "source_shadow_local": source_shadow_local,
        "matched": matched,
        "distractors": distractors,
    }


def _infer_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(colors) == 2, "expected exactly one body color and one shadow color"

    parses = []
    for body_color in colors:
        shadow_color = next(color for color in colors if color != body_color)
        parsed = _parse_scene(g, bg, body_color, shadow_color)
        if parsed is not None:
            parses.append(parsed)
    assert len(parses) == 1, "scene roles must be uniquely inferable"

    parsed = parses[0]
    parsed["bg"] = bg
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the exemplar sprite's attached shadow onto every rotated or mirrored copy in the same relative pose.

    Input structure:
    - The background is uniform.
    - One foreground color forms several separated 4-connected sprites; most are dihedral copies of one non-symmetric mother shape, while a few same-color distractor shapes are not.
    - Exactly one true copy has nearby pixels of a second foreground color attached as its shadow.

    Rule:
    - Identify the unique shadow-bearing exemplar and use its body mask as the reference orientation.
    - Find every same-color component congruent to that body mask under rotation and/or mirror, ignoring non-congruent distractors.
    - For each exemplar shadow pixel, take its offset in the exemplar's local sprite frame, apply the matched copy's same dihedral transform to that offset, and paint the shadow color there next to the copy.
    - Keep the exemplar, distractors, and all original body pixels unchanged.

    Key insight:
    - The shadow is not copied in absolute directions; it is attached to the sprite's local pose, so the shadow offsets must rotate or mirror together with each body copy.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _infer_scene(g)
    bg = int(parsed["bg"])
    shadow_color = int(parsed["shadow_color"])
    source_comp = parsed["source_comp"]
    source_mask = np.asarray(parsed["source_mask"], dtype=bool)
    source_shape = tuple(map(int, source_mask.shape))
    source_shadow_local = [tuple(map(int, rc)) for rc in parsed["source_shadow_local"]]

    out = g.copy()
    for item in parsed["matched"]:
        comp = item["comp"]
        if comp is source_comp:
            continue
        variant = item["variant"]
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        for sr, sc in source_shadow_local:
            tr, tc = _transform_point(sr, sc, source_shape, int(variant["k"]), bool(variant["mirrored"]))
            ar = r0 + tr
            ac = c0 + tc
            if 0 <= ar < out.shape[0] and 0 <= ac < out.shape[1] and int(out[ar, ac]) in {bg, shadow_color}:
                out[ar, ac] = shadow_color
    return out


def _tight_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _random_asymmetric_mask() -> np.ndarray:
    for _ in range(600):
        target_area = random.randint(6, 10)
        cells = {(0, 0)}
        while len(cells) < target_area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            nr, nc = r + dr, c + dc
            if abs(nr) > 5 or abs(nc) > 6:
                continue
            cells.add((nr, nc))
        mask = _tight_mask(cells)
        h, w = mask.shape
        if h < 3 or w < 3 or h > 6 or w > 7 or h == w:
            continue
        if int(mask.sum()) == h * w:
            continue
        if len(_dihedral_variants(mask)) != 8:
            continue
        return mask
    raise AssertionError("failed to sample an asymmetric source mask")


def _halo_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    body = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    out: list[tuple[int, int]] = []
    for r in range(-1, h + 1):
        for c in range(-1, w + 1):
            if (r, c) in body:
                continue
            if any(max(abs(r - br), abs(c - bc)) == 1 for br, bc in body):
                out.append((int(r), int(c)))
    return out


def _choose_shadow_local(
    source_mask: np.ndarray, target_variants: list[dict[str, object]]
) -> list[tuple[int, int]]:
    candidates = _halo_positions(source_mask)
    body = {tuple(map(int, rc)) for rc in np.argwhere(source_mask)}
    source_shape = tuple(map(int, source_mask.shape))
    for _ in range(500):
        count = random.randint(3, 4)
        chosen = random.sample(candidates, count)
        has_orth = any(any(abs(r - br) + abs(c - bc) == 1 for br, bc in body) for r, c in chosen)
        has_diag = any(any(max(abs(r - br), abs(c - bc)) == 1 and abs(r - br) + abs(c - bc) == 2 for br, bc in body) for r, c in chosen)
        if not (has_orth and has_diag):
            continue
        if len({r for r, _c in chosen}) < 2 or len({c for _r, c in chosen}) < 2:
            continue
        if all(
            sorted(chosen)
            == sorted(
                _transform_point(r, c, source_shape, int(variant["k"]), bool(variant["mirrored"]))
                for r, c in chosen
            )
            for variant in target_variants
        ):
            continue
        return [(int(r), int(c)) for r, c in chosen]
    raise AssertionError("failed to choose informative shadow offsets")


def _choose_target_variants(source_mask: np.ndarray) -> list[dict[str, object]]:
    variants = [variant for variant in _dihedral_variants(source_mask) if not (int(variant["k"]) == 0 and not bool(variant["mirrored"]))]
    quarter_turn = [variant for variant in variants if not bool(variant["mirrored"]) and int(variant["k"]) in {1, 3}]
    mirrored = [variant for variant in variants if bool(variant["mirrored"])]
    assert quarter_turn and mirrored

    target_count = random.randint(3, 5)
    chosen = [random.choice(quarter_turn), random.choice(mirrored)]
    used = {str(chosen[0]["name"]), str(chosen[1]["name"])}
    remaining = [variant for variant in variants if str(variant["name"]) not in used]
    random.shuffle(remaining)
    for variant in remaining:
        if len(chosen) >= target_count:
            break
        chosen.append(variant)
    random.shuffle(chosen)
    return chosen


def _random_distractor_mask(
    source_mask: np.ndarray, used_keys: set[tuple[tuple[int, int], bytes]], *, same_area: bool
) -> np.ndarray:
    src_area = int(source_mask.sum())
    src_h, src_w = map(int, source_mask.shape)
    for _ in range(500):
        area = src_area if same_area else max(4, src_area + random.choice([-2, -1, 1, 2]))
        cells = {(0, 0)}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            nr, nc = r + dr, c + dc
            if abs(nr) > 5 or abs(nc) > 6:
                continue
            cells.add((nr, nc))
        mask = _tight_mask(cells)
        h, w = mask.shape
        if h < 2 or w < 2 or h > 7 or w > 8:
            continue
        if same_area and area != int(mask.sum()):
            continue
        if _shape_key(mask) in used_keys:
            continue
        if any(
            variant["mask"].shape == mask.shape and np.array_equal(variant["mask"], mask)
            for variant in _dihedral_variants(source_mask)
        ):
            continue
        if h == src_h and w == src_w:
            return mask
        if same_area:
            return mask
        if int(mask.sum()) == src_area:
            return mask
        return mask
    raise AssertionError("failed to sample a distractor mask")


def _cell_bbox(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c, _v in cells]
    cols = [c for _r, c, _v in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _expand(cells: set[tuple[int, int]], gap: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def _make_item(
    kind: str,
    mask: np.ndarray,
    body_color: int,
    shadow_color: int,
    shadow_local: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    body_coords = [(int(r), int(c), int(body_color)) for r, c in np.argwhere(mask)]
    output_cells = body_coords.copy()
    input_cells = body_coords.copy()
    if shadow_local is not None:
        shadow_cells = [(int(r), int(c), int(shadow_color)) for r, c in shadow_local]
        if kind == "source":
            input_cells += shadow_cells
        output_cells += shadow_cells
    return {
        "kind": kind,
        "mask": mask,
        "input_cells": input_cells,
        "output_cells": output_cells,
    }


def _place_items(items: list[dict[str, object]]) -> tuple[int, int, dict[int, tuple[int, int]]]:
    items = sorted(
        items,
        key=lambda item: (
            (_cell_bbox(item["output_cells"])[2] - _cell_bbox(item["output_cells"])[0] + 1)
            * (_cell_bbox(item["output_cells"])[3] - _cell_bbox(item["output_cells"])[1] + 1)
        ),
        reverse=True,
    )
    for _ in range(400):
        max_h = max(_cell_bbox(item["output_cells"])[2] - _cell_bbox(item["output_cells"])[0] + 1 for item in items)
        max_w = max(_cell_bbox(item["output_cells"])[3] - _cell_bbox(item["output_cells"])[1] + 1 for item in items)
        min_dim = max(16, max(max_h, max_w) + 6)
        h = random.randint(min_dim, 30)
        w = random.randint(min_dim, 30)
        forbidden: set[tuple[int, int]] = set()
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for idx, item in enumerate(items):
            r0_min, c0_min, r0_max, c0_max = _cell_bbox(item["output_cells"])
            valid: list[tuple[int, int]] = []
            row_lo = 1 - r0_min
            row_hi = h - 2 - r0_max
            col_lo = 1 - c0_min
            col_hi = w - 2 - c0_max
            if row_lo > row_hi or col_lo > col_hi:
                ok = False
                break
            for origin_r in range(row_lo, row_hi + 1):
                for origin_c in range(col_lo, col_hi + 1):
                    abs_cells = {(origin_r + r, origin_c + c) for r, c, _v in item["output_cells"]}
                    if abs_cells & forbidden:
                        continue
                    valid.append((int(origin_r), int(origin_c)))
            if not valid:
                ok = False
                break
            origin = random.choice(valid)
            placements[id(item)] = origin
            abs_cells = {
                (origin[0] + r, origin[1] + c)
                for r, c, _v in item["output_cells"]
            }
            forbidden |= _expand(abs_cells, gap=1)
        if ok:
            return h, w, placements
    raise AssertionError("failed to place all items")


def _render_case(
    h: int, w: int, bg: int, items: list[dict[str, object]], placements: dict[int, tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    for item in items:
        origin_r, origin_c = placements[id(item)]
        for r, c, color in item["output_cells"]:
            out[origin_r + r, origin_c + c] = int(color)
        for r, c, color in item["input_cells"]:
            inp[origin_r + r, origin_c + c] = int(color)
    return inp, out


def _naive_absolute_shadow(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    parsed = _infer_scene(g)
    bg = int(parsed["bg"])
    shadow_color = int(parsed["shadow_color"])
    source_comp = parsed["source_comp"]
    sr0, sc0, _sr1, _sc1 = map(int, source_comp["bbox"])
    raw_offsets = [(int(r), int(c)) for r, c in parsed["source_shadow_local"]]
    out = g.copy()
    for item in parsed["matched"]:
        comp = item["comp"]
        if comp is source_comp:
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        for rr, cc in raw_offsets:
            ar = r0 + rr
            ac = c0 + cc
            if 0 <= ar < out.shape[0] and 0 <= ac < out.shape[1] and int(out[ar, ac]) == bg:
                out[ar, ac] = shadow_color
    return out


def _naive_shadow_all_bodies(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    parsed = _infer_scene(g)
    bg = int(parsed["bg"])
    shadow_color = int(parsed["shadow_color"])
    source_comp = parsed["source_comp"]
    source_mask = np.asarray(parsed["source_mask"], dtype=bool)
    source_shape = tuple(map(int, source_mask.shape))
    out = g.copy()
    for comp in parsed["components"]:
        if comp is source_comp:
            continue
        target_mask = _component_mask(g, comp)
        match = _match_variant(source_mask, target_mask)
        if match is None:
            match = {"k": 0, "mirrored": False}
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        for sr, sc in parsed["source_shadow_local"]:
            tr, tc = _transform_point(sr, sc, source_shape, int(match["k"]), bool(match["mirrored"]))
            ar = r0 + tr
            ac = c0 + tc
            if 0 <= ar < out.shape[0] and 0 <= ac < out.shape[1] and int(out[ar, ac]) == bg:
                out[ar, ac] = shadow_color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, body_color, shadow_color = random.sample(range(10), 3)
    base_mask = _random_asymmetric_mask()
    source_seed = random.choice(_dihedral_variants(base_mask))
    source_mask = np.asarray(source_seed["mask"], dtype=bool)
    target_variants = _choose_target_variants(source_mask)
    shadow_local = _choose_shadow_local(source_mask, target_variants)

    used_keys = {_shape_key(source_mask)}
    distractors = [
        _random_distractor_mask(source_mask, used_keys, same_area=True),
    ]
    used_keys.add(_shape_key(distractors[0]))
    if random.random() < 0.6:
        distractor2 = _random_distractor_mask(source_mask, used_keys, same_area=False)
        used_keys.add(_shape_key(distractor2))
        distractors.append(distractor2)

    items: list[dict[str, object]] = []
    items.append(_make_item("source", source_mask, body_color, shadow_color, shadow_local=shadow_local))
    for variant in target_variants:
        target_mask = np.asarray(variant["mask"], dtype=bool)
        transformed_shadow = [
            _transform_point(r, c, tuple(map(int, source_mask.shape)), int(variant["k"]), bool(variant["mirrored"]))
            for r, c in shadow_local
        ]
        items.append(
            _make_item(
                "copy",
                target_mask,
                body_color,
                shadow_color,
                shadow_local=[(int(r), int(c)) for r, c in transformed_shadow],
            )
        )
    for mask in distractors:
        items.append(_make_item("distractor", mask, body_color, shadow_color, shadow_local=None))

    try:
        h, w, placements = _place_items(items)
    except AssertionError:
        return None
    inp, expected = _render_case(h, w, bg, items, placements)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    try:
        parsed = _infer_scene(inp)
    except AssertionError:
        return None
    if len(parsed["distractors"]) != len(distractors):
        return None
    if len(parsed["matched"]) != 1 + len(target_variants):
        return None
    if np.array_equal(_naive_absolute_shadow(inp), expected):
        return None
    if np.array_equal(_naive_shadow_all_bodies(inp), expected):
        return None

    shadow_count = len(parsed["source_shadow_local"])
    if np.count_nonzero(expected == shadow_color) != np.count_nonzero(inp == shadow_color) + shadow_count * len(target_variants):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid shadow-covariance puzzle")

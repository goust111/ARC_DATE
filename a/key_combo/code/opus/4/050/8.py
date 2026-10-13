import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_BG = 0
_GREY = 5
_REF_PALETTE = (1, 2, 3)
_FREE_PALETTE = (4, 6, 7, 8, 9)
_MAX_ATTEMPTS = 400
_CONTROL_MIN_GAP = 5
_PLACEMENT_PAD = 1
_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


try:
    _arc_utils = SourceFileLoader("_arc_utils_opus4_050_8", str(_ARC_UTILS_PATH)).load_module()
    bbox_coords = _arc_utils.bbox_coords
    crop_bbox = _arc_utils.crop_bbox
    components_mask = _arc_utils.components_mask
except Exception:
    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        if connectivity == 4:
            steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
        else:
            steps = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
        comps = []
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
                    for dr, dc in steps:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.asarray(coords, dtype=int))
        return comps


def _normalize_shape(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0, keepdims=True)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), _BG, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _frame_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    sprite = _shape_patch(shape, color)
    h, w = sprite.shape
    patch = np.full((h + 2, w + 2), _BG, dtype=int)
    patch[0, :] = _GREY
    patch[-1, :] = _GREY
    patch[:, 0] = _GREY
    patch[:, -1] = _GREY
    patch[1:-1, 1:-1] = sprite
    return patch


def _rectangular_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    return out


def _extract_reference_entries(grid: np.ndarray) -> list[dict]:
    refs = []
    grey_comps = components_mask(grid == _GREY, connectivity=4)
    for coords in grey_comps:
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = _rectangular_border_mask(grid.shape, bbox)
        if coords.shape[0] != int(border.sum()):
            continue
        if not np.all(grid[border] == _GREY):
            continue
        patch = crop_bbox(grid, bbox)
        patch_border = np.zeros_like(patch, dtype=bool)
        patch_border[0, :] = True
        patch_border[-1, :] = True
        patch_border[:, 0] = True
        patch_border[:, -1] = True
        if not np.all(patch[patch_border] == _GREY):
            continue
        if np.any(patch[1:-1, 1:-1] == _GREY):
            continue
        inner = patch[1:-1, 1:-1]
        sprite_mask = inner != _BG
        sprite_comps = components_mask(sprite_mask, connectivity=4)
        if len(sprite_comps) != 1:
            continue
        sprite_coords = sprite_comps[0]
        colors = set(int(inner[r, c]) for r, c in sprite_coords)
        if len(colors) != 1:
            continue
        refs.append(
            {
                "bbox": bbox,
                "shape": _normalize_shape(sprite_coords),
                "color": next(iter(colors)),
            }
        )
    refs.sort(key=lambda item: item["bbox"])
    return refs


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    refs = _extract_reference_entries(g)
    assert 2 <= len(refs) <= 3, f"expected 2-3 reference frames, got {len(refs)}"

    shape_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    for ref in refs:
        shape = ref["shape"]
        color = int(ref["color"])
        assert color in _REF_PALETTE, f"reference color must be in {_REF_PALETTE}, got {color}"
        assert shape not in shape_to_color, "reference shapes must be unique"
        shape_to_color[shape] = color

    controls = {int(c): int(g[0, c]) for c in range(w) if int(g[0, c]) != _BG}
    assert 3 <= len(controls) <= 5, f"expected 3-5 top-row control pixels, got {len(controls)}"
    assert all(color in set(shape_to_color.values()) for color in controls.values()), "control colors must come from references"

    masked = g.copy()
    masked[0, :] = _BG
    for ref in refs:
        r0, c0, r1, c1 = ref["bbox"]
        masked[r0 : r1 + 1, c0 : c1 + 1] = _BG

    free_mask = (masked != _BG) & (masked != _GREY)
    free_infos = []
    for coords in components_mask(free_mask, connectivity=4):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        sprite_colors = set(int(g[r, c]) for r, c in coords)
        assert len(sprite_colors) == 1, "each free sprite must be monochromatic"
        permit_matches = [color for col, color in controls.items() if c0 <= col <= c1]
        assert len(permit_matches) <= 1, "each free sprite span may contain at most one control pixel"
        dict_color = shape_to_color.get(_normalize_shape(coords))
        permit_color = permit_matches[0] if permit_matches else None
        free_infos.append(
            {
                "coords": coords,
                "bbox": bbox,
                "shape": _normalize_shape(coords),
                "input_color": next(iter(sprite_colors)),
                "dict_color": dict_color,
                "permit_color": permit_color,
                "survives": dict_color is not None and permit_color is not None and dict_color == permit_color,
            }
        )

    free_infos.sort(key=lambda item: item["bbox"])
    assert 4 <= len(free_infos) <= 6, f"expected 4-6 free sprites, got {len(free_infos)}"
    return {"refs": refs, "shape_to_color": shape_to_color, "controls": controls, "free_infos": free_infos}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the free sprites whose reference-frame shape color agrees with their top-row column permission.

    Input structure:
    - The background is black, with 2-3 grey one-cell-thick hollow reference frames in the upper area.
    - Each frame encloses one monochrome 4-connected sprite, and the top row contains 3-5 colored singleton control pixels.
    - The lower area contains 4-6 monochrome free sprites; some share a frame sprite's exact bbox-relative shape and others do not.

    Rule:
    - Build a dictionary from each framed sprite's exact normalized shape to that sprite's color.
    - For each free sprite, look up its normalized shape in that dictionary and independently find the unique top-row control pixel whose column lies inside the free sprite's bbox column span.
    - If both colors exist and are equal, recolor the whole free sprite to that shared color; otherwise erase the free sprite to black. Keep the frames, framed sprites, and top-row controls unchanged.

    Key insight:
    - Survival is gated by two independent tests at once: shape matching supplies the candidate color, while column containment supplies the permission color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    analysis = _analyze_grid(g)
    out = g.copy()
    for info in analysis["free_infos"]:
        coords = info["coords"]
        rr = coords[:, 0]
        cc = coords[:, 1]
        if info["survives"]:
            out[rr, cc] = int(info["dict_color"])
        else:
            out[rr, cc] = _BG
    return out


def _rect_ok(rect: tuple[int, int, int, int], occupied: list[tuple[int, int, int, int]], pad: int = _PLACEMENT_PAD) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in occupied:
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != _BG
    region[mask] = patch[mask]


def _dedupe_shapes(raw_shapes: list[tuple[tuple[int, int], ...]]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    for shape in raw_shapes:
        norm = _normalize_shape(list(shape))
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


_REFERENCE_SHAPES = _dedupe_shapes(
    [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ]
)

_FREE_SHAPES = _dedupe_shapes(
    list(_REFERENCE_SHAPES)
    + [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
        ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)),
    ]
)


def _choose_spaced_columns(width: int, count: int, gap: int) -> list[int]:
    usable = width - 2
    slack = usable - (count - 1) * (gap - 1)
    if slack < count:
        raise RuntimeError("grid too narrow for spaced controls")
    bases = sorted(random.sample(range(slack), count))
    return [1 + base + idx * (gap - 1) for idx, base in enumerate(bases)]


def _choose_free_colors(count: int) -> list[int]:
    colors = list(_FREE_PALETTE)
    random.shuffle(colors)
    while len(colors) < count:
        extra = list(_FREE_PALETTE)
        random.shuffle(extra)
        colors.extend(extra)
    return colors[:count]


def _build_specs(refs: list[dict], free_count: int) -> list[dict]:
    ref_shapes = {ref["shape"] for ref in refs}
    ref_colors = [int(ref["color"]) for ref in refs]
    witness = random.choice(refs)
    witness_color = int(witness["color"])
    other_color = random.choice([color for color in ref_colors if color != witness_color])
    nonmatch_pool = [shape for shape in _FREE_SHAPES if shape not in ref_shapes]
    witness_nonmatch = random.choice(nonmatch_pool)

    specs = [
        {"shape": witness["shape"], "permit": witness_color, "role": "survive"},
        {"shape": witness["shape"], "permit": other_color, "role": "mismatch"},
        {"shape": witness_nonmatch, "permit": witness_color, "role": "nomatch_control"},
    ]

    optional_modes = (
        "survive",
        "mismatch",
        "match_none",
        "nomatch_control",
        "nomatch_none",
    )
    while len(specs) < free_count:
        mode = random.choice(optional_modes)
        ref = random.choice(refs)
        ref_color = int(ref["color"])
        if mode == "survive":
            specs.append({"shape": ref["shape"], "permit": ref_color, "role": mode})
        elif mode == "mismatch":
            wrong = random.choice([color for color in ref_colors if color != ref_color])
            specs.append({"shape": ref["shape"], "permit": wrong, "role": mode})
        elif mode == "match_none":
            specs.append({"shape": ref["shape"], "permit": None, "role": mode})
        elif mode == "nomatch_control":
            shape = random.choice(nonmatch_pool)
            specs.append({"shape": shape, "permit": random.choice(ref_colors), "role": mode})
        else:
            specs.append({"shape": random.choice(nonmatch_pool), "permit": None, "role": mode})

    colors = _choose_free_colors(len(specs))
    random.shuffle(specs)
    for spec, color in zip(specs, colors):
        spec["input_color"] = color
    return specs


def _valid_left_positions(width: int, sprite_w: int, controls: dict[int, int], permit: int | None) -> list[int]:
    control_cols = sorted(controls)
    valids = []
    for left in range(1, width - sprite_w):
        right = left + sprite_w - 1
        covered = [col for col in control_cols if left <= col <= right]
        if permit is None:
            if not covered:
                valids.append(left)
        else:
            if len(covered) == 1 and controls[covered[0]] == permit:
                valids.append(left)
    return valids


def _place_references(grid: np.ndarray, refs: list[dict]) -> tuple[list[dict], list[tuple[int, int, int, int]]]:
    h, w = grid.shape
    occupied: list[tuple[int, int, int, int]] = []
    placed = []
    for ref in refs:
        patch = _frame_patch(ref["shape"], ref["color"])
        ph, pw = patch.shape
        row_max = min(8, h // 2 - 3)
        if row_max < 2:
            raise RuntimeError("grid too short for references")
        for _ in range(200):
            top = random.randint(2, row_max)
            left = random.randint(1, w - pw - 1)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if _rect_ok(rect, occupied):
                _paste_patch(grid, patch, top, left)
                occupied.append(rect)
                placed.append({"shape": ref["shape"], "color": ref["color"], "bbox": rect})
                break
        else:
            raise RuntimeError("failed to place references")
    return placed, occupied


def _place_free_sprites(
    grid: np.ndarray,
    specs: list[dict],
    controls: dict[int, int],
    occupied: list[tuple[int, int, int, int]],
    free_row_min: int,
) -> None:
    h, w = grid.shape
    order = sorted(
        range(len(specs)),
        key=lambda idx: (-len(specs[idx]["shape"]), -_shape_dims(specs[idx]["shape"])[0], random.random()),
    )
    for idx in order:
        spec = specs[idx]
        patch = _shape_patch(spec["shape"], spec["input_color"])
        ph, pw = patch.shape
        valids = _valid_left_positions(w, pw, controls, spec["permit"])
        random.shuffle(valids)
        if not valids:
            raise RuntimeError("failed to find valid control span for free sprite")
        rows = list(range(free_row_min, h - ph + 1))
        random.shuffle(rows)
        for left in valids:
            for top in rows:
                rect = (top, left, top + ph - 1, left + pw - 1)
                if _rect_ok(rect, occupied):
                    _paste_patch(grid, patch, top, left)
                    occupied.append(rect)
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("failed to place free sprite")


def _validate_generated(grid: np.ndarray) -> None:
    analysis = _analyze_grid(grid)
    out = main(grid)
    assert np.array_equal(out[0, :], grid[0, :]), "top-row controls must stay unchanged"
    for ref in analysis["refs"]:
        r0, c0, r1, c1 = ref["bbox"]
        assert np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], grid[r0 : r1 + 1, c0 : c1 + 1]), (
            "reference frames and enclosed sprites must stay unchanged"
        )

    free_infos = analysis["free_infos"]
    survivor_count = 0
    erased_count = 0
    mismatch_seen = False
    nomatch_seen = False
    same_shape_split = False
    same_permission_split = False

    for info in free_infos:
        rr = info["coords"][:, 0]
        cc = info["coords"][:, 1]
        if info["survives"]:
            survivor_count += 1
            assert np.all(out[rr, cc] == int(info["dict_color"]))
        else:
            erased_count += 1
            assert np.all(out[rr, cc] == _BG)
        if info["dict_color"] is not None and info["permit_color"] is not None and info["dict_color"] != info["permit_color"]:
            mismatch_seen = True
        if info["dict_color"] is None:
            nomatch_seen = True

    for a in free_infos:
        for b in free_infos:
            if a is b:
                continue
            if a["survives"] and a["shape"] == b["shape"] and b["dict_color"] == a["dict_color"] and b["permit_color"] not in (None, a["dict_color"]):
                same_shape_split = True
            if a["survives"] and b["dict_color"] is None and a["permit_color"] == b["permit_color"] and a["permit_color"] is not None:
                same_permission_split = True

    assert survivor_count >= 1, "need at least one surviving free sprite"
    assert erased_count >= 1, "need at least one erased free sprite"
    assert mismatch_seen, "need a shape match with mismatching permission color"
    assert nomatch_seen, "need a free sprite with no dictionary match"
    assert same_shape_split, "need the same shape to produce different outcomes under different permissions"
    assert same_permission_split, "need the same permission color to allow one match and reject one non-match"


def _gen_attempt() -> np.ndarray:
    height = random.randint(20, 26)
    width = random.randint(23, 28)
    grid = np.full((height, width), _BG, dtype=int)

    ref_count = random.randint(2, 3)
    control_count = random.randint(max(3, ref_count), 5)
    free_count = random.randint(4, 6)

    ref_shapes = random.sample(_REFERENCE_SHAPES, ref_count)
    ref_colors = random.sample(list(_REF_PALETTE), ref_count)
    ref_templates = [{"shape": shape, "color": color} for shape, color in zip(ref_shapes, ref_colors)]

    placed_refs, occupied = _place_references(grid, ref_templates)
    control_cols = _choose_spaced_columns(width, control_count, _CONTROL_MIN_GAP)
    control_colors = list(ref_colors)
    while len(control_colors) < control_count:
        control_colors.append(random.choice(ref_colors))
    random.shuffle(control_colors)
    controls = {col: color for col, color in zip(control_cols, control_colors)}
    for col, color in controls.items():
        grid[0, col] = color

    free_row_min = max(ref["bbox"][2] for ref in placed_refs) + 2
    if free_row_min > height - 4:
        raise RuntimeError("not enough room below references")

    specs = _build_specs(placed_refs, free_count)
    _place_free_sprites(grid, specs, controls, occupied, free_row_min)
    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        try:
            return _gen_attempt()
        except RuntimeError:
            continue
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 8")

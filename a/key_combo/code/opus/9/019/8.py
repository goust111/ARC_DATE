from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("arc_utils_shared", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ARC = _load_arc_utils()


if _ARC is not None:
    crop_bbox = _ARC.crop_bbox
    find_components = _ARC.find_components
    get_bg = _ARC.get_bg
    is_contiguous = _ARC.is_contiguous
    paste = _ARC.paste
else:
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        h, w = patch.shape
        rr0 = int(r0)
        cc0 = int(c0)
        rr1 = rr0 + h
        cc1 = cc0 + w
        if not clip:
            if rr0 < 0 or cc0 < 0 or rr1 > dst.shape[0] or cc1 > dst.shape[1]:
                raise ValueError("paste out of bounds")
            patch_view = patch
            dst_view = dst[rr0:rr1, cc0:cc1]
        else:
            rr0_clip = max(0, rr0)
            cc0_clip = max(0, cc0)
            rr1_clip = min(dst.shape[0], rr1)
            cc1_clip = min(dst.shape[1], cc1)
            if rr0_clip >= rr1_clip or cc0_clip >= cc1_clip:
                return dst
            pr0 = rr0_clip - rr0
            pc0 = cc0_clip - cc0
            pr1 = pr0 + (rr1_clip - rr0_clip)
            pc1 = pc0 + (cc1_clip - cc0_clip)
            patch_view = patch[pr0:pr1, pc0:pc1]
            dst_view = dst[rr0_clip:rr1_clip, cc0_clip:cc1_clip]
        if transparent is None:
            dst_view[:, :] = patch_view
        else:
            mask = patch_view != int(transparent)
            dst_view[mask] = patch_view[mask]
        return dst

    def is_contiguous(bitmask: np.ndarray, background: int = 0, connectivity: int = 4) -> bool:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        coords = np.argwhere(bitmask != background)
        if coords.size == 0:
            return False
        seen = set()
        stack = [tuple(map(int, coords[0]))]
        seen.add(stack[0])
        h, w = bitmask.shape
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and bitmask[nr, nc] != background and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return len(seen) == int((bitmask != background).sum())

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        return_format: str = "dict",
    ) -> list[dict]:
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        h, w = grid.shape
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        visited = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                if visited[r, c] or color == bg:
                    continue
                visited[r, c] = True
                stack = [(r, c)]
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if not (0 <= nr < h and 0 <= nc < w) or visited[nr, nc]:
                            continue
                        if monochromatic:
                            ok = int(grid[nr, nc]) == color
                        else:
                            ok = int(grid[nr, nc]) != bg
                        if ok:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                arr = np.array(coords, dtype=int)
                out.append(
                    {
                        "color": color if monochromatic else None,
                        "coords": arr,
                        "bbox": (
                            int(arr[:, 0].min()),
                            int(arr[:, 1].min()),
                            int(arr[:, 0].max()),
                            int(arr[:, 1].max()),
                        ),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


BG = 0
SEP = 5
FG_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500

_SHAPE_MASKS_RAW = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
]

_GLYPH_MASKS = [
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=bool,
    ),
]

_COUNT_MULTISETS = [
    (1, 2, 2, 3),
    (1, 1, 2, 3),
    (1, 1, 2, 4),
    (1, 1, 3, 3),
    (1, 2, 3, 3),
    (1, 1, 2, 2, 3),
    (1, 1, 2, 2, 4),
    (2, 2, 2),
    (2, 2, 3),
]


def canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    base = np.asarray(mask, dtype=bool)
    best = (base.tobytes(), tuple(map(int, base.shape)))
    cur = base
    for _ in range(4):
        for variant in (cur, np.fliplr(cur)):
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if key < best:
                best = key
        cur = np.rot90(cur)
    return best


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen = set()
    out = []
    cur = np.asarray(mask, dtype=bool)
    for _ in range(4):
        for variant in (cur, np.fliplr(cur)):
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if key not in seen:
                seen.add(key)
                out.append(variant.copy())
        cur = np.rot90(cur)
    return out


def _shape_library() -> list[dict]:
    by_key: dict[tuple[bytes, tuple[int, int]], dict] = {}
    for mask in _SHAPE_MASKS_RAW:
        key = canonical_shape_key(mask)
        if key in by_key:
            continue
        variants = _dihedral_variants(mask)
        by_key[key] = {
            "mask": min(variants, key=lambda arr: (arr.shape, arr.tobytes())),
            "key": key,
            "area": int(mask.sum()),
            "variants": variants,
        }
    return list(by_key.values())


_SHAPE_LIBRARY = _shape_library()


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _separator_row(grid: np.ndarray) -> int:
    candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert len(candidates) == 1, "expected one full-width grey separator row"
    return int(candidates[0])


def _parse_legend(legend: np.ndarray) -> dict[int, np.ndarray]:
    items = []
    for comp in find_components(legend, bg=BG, connectivity=4, monochromatic=False, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
            continue
        assert c1 + 3 < legend.shape[1], "marker slot must fit to the right of glyph"
        assert np.all(legend[r0 : r1 + 1, c1 + 1] == BG), "glyph-marker gap must be blank"
        glyph = crop_bbox(legend, (r0, c0, r1, c1))
        marker = legend[r0 : r1 + 1, c1 + 2 : c1 + 4]
        count = int(np.count_nonzero(marker != BG))
        assert 1 <= count <= 4, "marker count must be between 1 and 4"
        items.append((count, glyph, r0, c0))
    items.sort(key=lambda item: (item[2], item[3]))
    mapping: dict[int, np.ndarray] = {}
    for count, glyph, _r0, _c0 in items:
        assert count not in mapping, "legend frequency keys must be unique"
        mapping[count] = glyph
    assert 2 <= len(mapping) <= 4, "legend must provide 2 to 4 glyphs"
    return mapping


def _extract_main_objects(grid: np.ndarray, sep_row: int) -> list[dict]:
    upper = grid[:sep_row, :]
    objects = []
    for comp in find_components(upper, bg=BG, connectivity=4, monochromatic=True, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(upper, (r0, c0, r1, c1))
        mask = patch != BG
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": [tuple(map(int, coord)) for coord in comp["coords"]],
                "mask": mask,
                "shape_key": canonical_shape_key(mask),
            }
        )
    assert 6 <= len(objects) <= 10, "expected 6 to 10 foreground objects in the main region"
    return objects


def _render_from_parsed(grid: np.ndarray, freq_to_glyph: dict[int, np.ndarray], objects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    sep_row = _separator_row(out)
    out[sep_row:, :] = BG

    freq = Counter(obj["shape_key"] for obj in objects)
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BG

    for obj in sorted(objects, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        r0, c0, _r1, _c1 = obj["bbox"]
        glyph = freq_to_glyph[int(freq[obj["shape_key"]])]
        paste(out, glyph, r0, c0, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each main-region object with the glyph indexed by that object's rotation/flip-invariant shape frequency.

    Input structure:
    - A black background grid is split by one full-width grey horizontal separator.
    - Above the separator are 6 to 10 small monochrome 4-connected objects; different objects may use different colors.
    - Below the separator is a legend made of 2 to 4 items, each consisting of a 4x4 glyph and a disconnected dot marker one column to its right.

    Rule:
    - Read the legend as frequency -> glyph: the number of non-black pixels in an item's marker gives the key, and the 4x4 patch to its left is the glyph value.
    - In the main region, crop every object to its bbox, convert it to a binary mask, canonicalize that mask over the 8 rotation/flip variants, and count how many objects share each canonical shape.
    - Clear the original objects, then for each object stamp the glyph for its frequency at the object's bbox top-left corner; later objects in reading order overwrite earlier stamps on overlap.
    - Remove the separator and the entire legend region to black in the output.

    Key insight:
    - Shape appearance is only an intermediate code: objects are grouped by invariant shape, but the visible replacement is chosen solely by the global frequency of that shape class.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    sep_row = _separator_row(g)
    legend = g[sep_row + 1 :, :]
    mapping = _parse_legend(legend)
    objects = _extract_main_objects(g, sep_row)
    freq = Counter(obj["shape_key"] for obj in objects)
    assert set(freq.values()) <= set(mapping.keys()), "every observed frequency must exist in the legend"
    return _render_from_parsed(g, mapping, objects)


def _marker_pattern(count: int, rng: np.random.Generator) -> np.ndarray:
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(cells)
    marker = np.zeros((4, 2), dtype=bool)
    for r, c in cells[:count]:
        marker[r + 1, c] = True
    return marker


def _choose_class_specs(rng: np.random.Generator) -> list[dict] | None:
    counts = list(_COUNT_MULTISETS[int(rng.integers(0, len(_COUNT_MULTISETS)))])
    rng.shuffle(counts)
    if len(counts) > len(_SHAPE_LIBRARY):
        return None
    for _ in range(200):
        shape_ids = rng.choice(len(_SHAPE_LIBRARY), size=len(counts), replace=False)
        specs = []
        for freq, shape_id in zip(counts, shape_ids):
            shape = _SHAPE_LIBRARY[int(shape_id)]
            specs.append(
                {
                    "freq": int(freq),
                    "mask": shape["mask"],
                    "key": shape["key"],
                    "area": int(shape["area"]),
                    "variants": [v.copy() for v in shape["variants"]],
                }
            )
        if not any(a["area"] == b["area"] and a["freq"] != b["freq"] for i, a in enumerate(specs) for b in specs[i + 1 :]):
            continue
        if not any(a["freq"] == b["freq"] and a["key"] != b["key"] for i, a in enumerate(specs) for b in specs[i + 1 :]):
            continue
        return specs
    return None


def _object_cells_valid(mask: np.ndarray, r0: int, c0: int, occupied: set[tuple[int, int]], h_lim: int, w_lim: int) -> bool:
    h, w = map(int, mask.shape)
    if r0 < 0 or c0 < 0 or r0 + h > h_lim or c0 + w > w_lim:
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _place_mask(mask: np.ndarray, r0: int, c0: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = _coords_from_mask(mask, r0, c0)
    occupied.update(cells)
    return cells


def _stamp_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return not (ar + 3 < br or br + 3 < ar or ac + 3 < bc or bc + 3 < ac)


def _find_overlap_partner(
    rng: np.random.Generator,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    first_anchor: tuple[int, int],
    occupied: set[tuple[int, int]],
    h_lim: int,
    w_lim: int,
) -> tuple[int, int] | None:
    fr, fc = first_anchor
    candidates = []
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            if dr == 0 and dc == 0:
                continue
            rr = int(fr + dr)
            cc = int(fc + dc)
            if not _stamp_overlap((fr, fc), (rr, cc)):
                continue
            if not _object_cells_valid(second_mask, rr, cc, occupied, h_lim, w_lim):
                continue
            overlap_r = max(0, min(fr + 3, rr + 3) - max(fr, rr) + 1)
            overlap_c = max(0, min(fc + 3, cc + 3) - max(fc, cc) + 1)
            overlap_area = overlap_r * overlap_c
            if 1 <= overlap_area <= 6:
                candidates.append((rr, cc))
    if not candidates:
        return None
    return tuple(map(int, candidates[int(rng.integers(0, len(candidates)))]))


def _all_valid_anchors(mask: np.ndarray, occupied: set[tuple[int, int]], h_lim: int, w_lim: int) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    out = []
    for r0 in range(h_lim - h + 1):
        for c0 in range(w_lim - w + 1):
            if _object_cells_valid(mask, r0, c0, occupied, h_lim, w_lim):
                out.append((int(r0), int(c0)))
    return out


def _build_instances(specs: list[dict], rng: np.random.Generator) -> list[dict]:
    instances = []
    for class_id, spec in enumerate(specs):
        variants = spec["variants"]
        chosen = []
        if spec["freq"] >= 2 and len(variants) >= 2:
            picks = rng.choice(len(variants), size=2, replace=False)
            chosen.extend([variants[int(picks[0])].copy(), variants[int(picks[1])].copy()])
        while len(chosen) < spec["freq"]:
            chosen.append(variants[int(rng.integers(0, len(variants)))].copy())
        rng.shuffle(chosen)
        for mask in chosen:
            instances.append(
                {
                    "class_id": int(class_id),
                    "freq": int(spec["freq"]),
                    "canonical_key": spec["key"],
                    "area": int(spec["area"]),
                    "mask": mask,
                }
            )
    return instances


def _assign_object_colors(instances: list[dict], rng: np.random.Generator) -> bool:
    palette = list(rng.choice(FG_COLORS, size=int(rng.integers(3, 6)), replace=False))
    for item in instances:
        item["color"] = int(palette[int(rng.integers(0, len(palette)))])
    color_to_freqs: dict[int, set[int]] = defaultdict(set)
    for item in instances:
        color_to_freqs[int(item["color"])].add(int(item["freq"]))
    return any(len(freqs) >= 2 for freqs in color_to_freqs.values())


def _render_legend(freq_to_glyph: dict[int, np.ndarray], sep_row: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    legend = np.zeros((h - sep_row - 1, w), dtype=int)
    keys = list(freq_to_glyph.keys())
    n = len(keys)
    if n == 4:
        rows, cols = 2, 2
    elif n == 3 and w >= 23 and rng.random() < 0.5:
        rows, cols = 1, 3
    elif n == 2 and rng.random() < 0.3:
        rows, cols = 2, 1
    else:
        rows = 1 if n <= 3 else 2
        cols = int(np.ceil(n / rows))
    item_h = 4
    item_w = 7
    row_gap = 1 if rows > 1 else 0
    col_gap = 1
    need_h = rows * item_h + (rows - 1) * row_gap
    need_w = cols * item_w + (cols - 1) * col_gap
    if need_h > legend.shape[0] or need_w > legend.shape[1]:
        return None

    row_slack = legend.shape[0] - need_h
    col_slack = legend.shape[1] - need_w
    top_margin = int(rng.integers(0, row_slack + 1)) if row_slack > 0 else 0
    left_margin = int(rng.integers(0, col_slack + 1)) if col_slack > 0 else 0
    row_positions = [top_margin + r * (item_h + row_gap) for r in range(rows)]
    col_positions = [left_margin + c * (item_w + col_gap) for c in range(cols)]

    order = keys.copy()
    rng.shuffle(order)
    slots = [(r, c) for r in row_positions for c in col_positions][:n]
    for freq, (r0, c0) in zip(order, slots):
        glyph = freq_to_glyph[int(freq)]
        marker = _marker_pattern(int(freq), rng)
        if np.any(legend[r0 : r0 + 4, c0 : c0 + 7] != BG):
            return None
        paste(legend, glyph, r0, c0)
        legend[r0 : r0 + 4, c0 + 5 : c0 + 7][marker] = glyph[glyph != BG][0]
    return legend


def _freq_to_glyphs(freq_keys: list[int], rng: np.random.Generator) -> dict[int, np.ndarray]:
    glyph_ids = rng.choice(len(_GLYPH_MASKS), size=len(freq_keys), replace=False)
    colors = list(rng.choice(FG_COLORS, size=len(freq_keys), replace=False))
    mapping = {}
    for freq, glyph_id, color in zip(freq_keys, glyph_ids, colors):
        mapping[int(freq)] = _mask_to_patch(_GLYPH_MASKS[int(glyph_id)], int(color))
    return mapping


def _alt_render_by_raw_shape(case: dict) -> np.ndarray | None:
    out = np.array(case["grid"], dtype=int)
    sep_row = int(case["sep_row"])
    out[sep_row:, :] = BG
    raw_counter = Counter((inst["mask"].tobytes(), tuple(map(int, inst["mask"].shape))) for inst in case["instances"])
    for inst in case["instances"]:
        for r, c in inst["coords"]:
            out[r, c] = BG
    for inst in sorted(case["instances"], key=lambda item: (item["anchor"][0], item["anchor"][1])):
        key = (inst["mask"].tobytes(), tuple(map(int, inst["mask"].shape)))
        freq = int(raw_counter[key])
        if freq not in case["freq_to_glyph"]:
            return None
        glyph = case["freq_to_glyph"][freq]
        paste(out, glyph, inst["anchor"][0], inst["anchor"][1], clip=True)
    return out


def _alt_render_reverse_order(case: dict) -> np.ndarray:
    out = np.array(case["grid"], dtype=int)
    sep_row = int(case["sep_row"])
    out[sep_row:, :] = BG
    for inst in case["instances"]:
        for r, c in inst["coords"]:
            out[r, c] = BG
    for inst in sorted(case["instances"], key=lambda item: (item["anchor"][0], item["anchor"][1]), reverse=True):
        paste(out, case["freq_to_glyph"][int(inst["freq"])], inst["anchor"][0], inst["anchor"][1], clip=True)
    return out


def _semantic_checks(case: dict) -> bool:
    grid = case["grid"]
    out = case["output"]
    if grid.shape[0] > 30 or grid.shape[1] > 30 or out.shape[0] > 30 or out.shape[1] > 30:
        return False
    parsed_out = main(grid)
    if not np.array_equal(parsed_out, out):
        return False
    sep_row = int(case["sep_row"])
    if np.any(out[sep_row:, :] != BG):
        return False
    freq_keys = set(case["freq_to_glyph"].keys())
    object_freqs = {int(spec["freq"]) for spec in case["class_specs"]}
    if object_freqs != freq_keys:
        return False
    if not any(a["area"] == b["area"] and a["freq"] != b["freq"] for i, a in enumerate(case["class_specs"]) for b in case["class_specs"][i + 1 :]):
        return False
    same_freq_pairs = [(a, b) for i, a in enumerate(case["class_specs"]) for b in case["class_specs"][i + 1 :] if a["freq"] == b["freq"]]
    if not same_freq_pairs:
        return False
    color_to_freqs: dict[int, set[int]] = defaultdict(set)
    for inst in case["instances"]:
        color_to_freqs[int(inst["color"])].add(int(inst["freq"]))
    if not any(len(freqs) >= 2 for freqs in color_to_freqs.values()):
        return False
    if not case["needs_canonicalization"]:
        return False
    raw_shape_out = _alt_render_by_raw_shape(case)
    if raw_shape_out is not None and np.array_equal(raw_shape_out, out):
        return False
    if case["has_overlap"] and np.array_equal(_alt_render_reverse_order(case), out):
        return False
    return True


def _build_case(rng: np.random.Generator) -> dict | None:
    specs = _choose_class_specs(rng)
    if specs is None:
        return None
    instances = _build_instances(specs, rng)
    if not (6 <= len(instances) <= 10):
        return None
    if not _assign_object_colors(instances, rng):
        return None

    freq_keys = sorted({int(spec["freq"]) for spec in specs})
    n_keys = len(freq_keys)
    if n_keys == 4:
        legend_rows = 9
        min_w = 15
    elif n_keys == 3:
        legend_rows = 4 if rng.random() < 0.5 else 9
        min_w = 23 if legend_rows == 4 else 15
    else:
        legend_rows = 4 if rng.random() < 0.7 else 9
        min_w = 15

    w = int(rng.integers(max(18, min_w), 27))
    sep_row = int(rng.integers(10, 14))
    h = sep_row + 1 + legend_rows + int(rng.integers(1, 3))
    if h > 30:
        return None
    main_h_limit = sep_row - 2
    if main_h_limit < 7:
        return None

    occupied: set[tuple[int, int]] = set()
    placed = []
    overlap_target = bool(rng.random() < 0.7)
    clip_target = bool(rng.random() < 0.45)

    place_order = sorted(range(len(instances)), key=lambda i: (instances[i]["mask"].shape[0] * instances[i]["mask"].shape[1], instances[i]["area"]), reverse=True)
    overlap_pair: tuple[int, int] | None = None
    if overlap_target and len(instances) >= 2:
        repeated = [i for i, item in enumerate(instances) if item["freq"] >= 2 and len(_dihedral_variants(item["mask"])) >= 1]
        pool = repeated if len(repeated) >= 2 else list(range(len(instances)))
        picks = rng.choice(pool, size=2, replace=False)
        overlap_pair = (int(picks[0]), int(picks[1]))

    if overlap_pair is not None:
        first, second = overlap_pair
        first_mask = instances[first]["mask"]
        valid_first = _all_valid_anchors(first_mask, occupied, main_h_limit, w)
        rng.shuffle(valid_first)
        paired = False
        for anchor in valid_first[:80]:
            if clip_target and anchor[1] <= w - 4 and first == place_order[0]:
                pass
            first_cells = _place_mask(first_mask, anchor[0], anchor[1], occupied)
            partner = _find_overlap_partner(rng, first_mask, instances[second]["mask"], anchor, occupied, main_h_limit, w)
            if partner is not None:
                second_cells = _place_mask(instances[second]["mask"], partner[0], partner[1], occupied)
                instances[first]["anchor"] = anchor
                instances[first]["coords"] = first_cells
                instances[second]["anchor"] = partner
                instances[second]["coords"] = second_cells
                placed.extend([first, second])
                paired = True
                break
            for cell in first_cells:
                occupied.remove(cell)
        if not paired:
            return None

    if clip_target:
        clip_candidates = [i for i in place_order if i not in placed]
        rng.shuffle(clip_candidates)
        clipped = False
        for idx in clip_candidates:
            mask = instances[idx]["mask"]
            h_mask, w_mask = map(int, mask.shape)
            anchors = []
            for c0 in range(max(0, w - 3), w - w_mask + 1):
                for r0 in range(main_h_limit - h_mask + 1):
                    if c0 <= w - 4:
                        continue
                    if _object_cells_valid(mask, r0, c0, occupied, main_h_limit, w):
                        anchors.append((r0, c0))
            if not anchors:
                continue
            anchor = anchors[int(rng.integers(0, len(anchors)))]
            cells = _place_mask(mask, anchor[0], anchor[1], occupied)
            instances[idx]["anchor"] = anchor
            instances[idx]["coords"] = cells
            placed.append(idx)
            clipped = True
            break
        if not clipped:
            return None

    remaining = [i for i in place_order if i not in placed]
    for idx in remaining:
        mask = instances[idx]["mask"]
        anchors = _all_valid_anchors(mask, occupied, main_h_limit, w)
        if not anchors:
            return None
        anchor = anchors[int(rng.integers(0, len(anchors)))]
        cells = _place_mask(mask, anchor[0], anchor[1], occupied)
        instances[idx]["anchor"] = anchor
        instances[idx]["coords"] = cells

    grid = np.zeros((h, w), dtype=int)
    for inst in instances:
        patch = _mask_to_patch(inst["mask"], inst["color"])
        paste(grid, patch, inst["anchor"][0], inst["anchor"][1], transparent=BG)
        inst["bbox"] = _bbox_from_coords(inst["coords"])

    freq_to_glyph = _freq_to_glyphs(freq_keys, rng)
    legend = _render_legend(freq_to_glyph, sep_row, h, w, rng)
    if legend is None:
        return None
    grid[sep_row, :] = SEP
    paste(grid, legend, sep_row + 1, 0)

    case = {
        "grid": grid,
        "sep_row": sep_row,
        "class_specs": specs,
        "instances": instances,
        "freq_to_glyph": freq_to_glyph,
        "has_overlap": any(
            _stamp_overlap(instances[i]["anchor"], instances[j]["anchor"])
            for i in range(len(instances))
            for j in range(i + 1, len(instances))
        ),
    }
    case["needs_canonicalization"] = any(
        len(
            {
                (inst["mask"].tobytes(), tuple(map(int, inst["mask"].shape)))
                for inst in instances
                if inst["class_id"] == class_id
            }
        )
        >= 2
        for class_id, spec in enumerate(specs)
        if spec["freq"] >= 2
    )
    parsed_objects = _extract_main_objects(grid, sep_row)
    parsed_freq = Counter(obj["shape_key"] for obj in parsed_objects)
    if not set(map(int, parsed_freq.values())) <= set(map(int, freq_to_glyph.keys())):
        return None
    case["output"] = _render_from_parsed(grid, freq_to_glyph, parsed_objects)
    if not _semantic_checks(case):
        return None
    return case


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        case = _build_case(rng)
        if case is not None:
            return case["grid"]
    raise RuntimeError("failed to generate a valid input")

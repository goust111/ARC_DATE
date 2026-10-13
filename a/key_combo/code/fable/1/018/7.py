from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

try:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color, bbox, crop_bbox, find_components, get_bg  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(np.asarray(mask, dtype=bool))
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
    ) -> list[dict]:
        assert monochromatic
        assert return_format == "dict"
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        if colors is None:
            color_list = sorted(set(map(int, g.flatten())) - {int(bg)})
        else:
            color_list = [int(c) for c in colors if int(c) != int(bg)]

        def neighbors(r: int, c: int):
            yield r - 1, c
            yield r + 1, c
            yield r, c - 1
            yield r, c + 1
            if int(connectivity) == 8:
                yield r - 1, c - 1
                yield r - 1, c + 1
                yield r + 1, c - 1
                yield r + 1, c + 1

        out: list[dict] = []
        for color in color_list:
            mask = g == int(color)
            seen = np.zeros(mask.shape, dtype=bool)
            for r in range(mask.shape[0]):
                for c in range(mask.shape[1]):
                    if not mask[r, c] or seen[r, c]:
                        continue
                    q: deque[tuple[int, int]] = deque([(r, c)])
                    seen[r, c] = True
                    coords = []
                    while q:
                        rr, cc = q.popleft()
                        coords.append((rr, cc))
                        for nr, nc in neighbors(rr, cc):
                            if (
                                0 <= nr < mask.shape[0]
                                and 0 <= nc < mask.shape[1]
                                and mask[nr, nc]
                                and not seen[nr, nc]
                            ):
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    coord_arr = np.array(coords, dtype=int)
                    out.append(
                        {
                            "color": int(color),
                            "coords": coord_arr,
                            "bbox": (0, 0, 0, 0),
                            "area": int(coord_arr.shape[0]),
                        }
                    )
                    r0, c0 = coord_arr.min(axis=0)
                    r1, c1 = coord_arr.max(axis=0)
                    out[-1]["bbox"] = (int(r0), int(c0), int(r1), int(c1))
        return out


BG = Color.BLACK
PANEL_H_RANGE = (6, 8)
PANEL_W_RANGE = (8, 10)
TOP_GAP_RANGE = (2, 4)
TARGET_SCALE_CHOICES = (2, 3)
OTHER_SCALE_CHOICES = (1, 2, 3)
NUM_CANDIDATES_RANGE = (4, 5)
DIFF_AREA_RANGE = (4, 8)
MAX_BASE_DIM = 6
PAIR_ATTEMPTS = 500
SHAPE_ATTEMPTS = 300
PLACE_ATTEMPTS = 300
GENERATE_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    box = bbox(np.asarray(mask, dtype=bool))
    if box is None:
        return None
    return crop_bbox(np.asarray(mask, dtype=bool), box).astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    bitmask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(bitmask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < bitmask.shape[0] and 0 <= nc < bitmask.shape[1] and bitmask[nr, nc]:
                p = (int(nr), int(nc))
                if p not in seen:
                    seen.add(p)
                    q.append(p)
    return len(seen) == int(bitmask.sum())


def _has_hole(mask: np.ndarray) -> bool:
    bitmask = np.asarray(mask, dtype=bool)
    h, w = bitmask.shape
    bg = ~bitmask
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any(bg & ~exterior))


def _dihedral(mask: np.ndarray, code: int) -> np.ndarray:
    out = np.rot90(np.asarray(mask, dtype=bool), int(code) % 4)
    if int(code) >= 4:
        out = np.fliplr(out)
    return out.copy()


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    tight = _crop_mask(mask)
    assert tight is not None
    out: list[np.ndarray] = []
    seen = set()
    for code in range(8):
        var = _dihedral(tight, code)
        key = (int(var.shape[0]), int(var.shape[1]), var.astype(np.uint8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(var)
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    keys = []
    for var in _dihedral_variants(mask):
        arr = var.astype(np.uint8)
        keys.append((int(arr.shape[0]), int(arr.shape[1]), arr.tobytes()))
    return min(keys)


def _valid_block_factors(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    bitmask = np.asarray(mask, dtype=bool)
    h, w = bitmask.shape
    valid: list[tuple[int, np.ndarray]] = []
    for k in range(1, min(h, w) + 1):
        if h % k or w % k:
            continue
        down = np.zeros((h // k, w // k), dtype=bool)
        ok = True
        for br in range(0, h, k):
            for bc in range(0, w, k):
                block = bitmask[br : br + k, bc : bc + k]
                if bool(block.all()):
                    down[br // k, bc // k] = True
                elif bool(block.any()):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        if not (down[0].any() and down[-1].any() and down[:, 0].any() and down[:, -1].any()):
            continue
        valid.append((int(k), down))
    return valid


def _infer_block_shape(mask: np.ndarray) -> tuple[int, np.ndarray]:
    valid = _valid_block_factors(mask)
    assert valid, "candidate must admit at least one block factor"
    k, down = max(valid, key=lambda item: item[0])
    return int(k), np.asarray(down, dtype=bool)


def _is_primitive(mask: np.ndarray) -> bool:
    return max(k for k, _ in _valid_block_factors(np.asarray(mask, dtype=bool))) == 1


def _interesting(mask: np.ndarray) -> bool:
    bitmask = np.asarray(mask, dtype=bool)
    h, w = bitmask.shape
    if h < 2 or w < 2:
        return False
    if int(bitmask.sum()) < 4:
        return False
    if int(bitmask.sum()) == h * w:
        return False
    row_keys = {row.astype(np.uint8).tobytes() for row in bitmask}
    col_keys = {col.astype(np.uint8).tobytes() for col in bitmask.T}
    return len(row_keys) >= 2 and len(col_keys) >= 2


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    if area <= 0 or area > height * width:
        return None
    start = (random.randrange(height), random.randrange(width))
    chosen = {start}
    frontier = set(_neighbors4(*start))
    while len(chosen) < area and frontier:
        candidates = [
            (r, c)
            for r, c in frontier
            if 0 <= r < height and 0 <= c < width and (r, c) not in chosen
        ]
        if not candidates:
            break
        cell = random.choice(candidates)
        chosen.add(cell)
        frontier.discard(cell)
        frontier.update(_neighbors4(*cell))
    if len(chosen) != area:
        return None
    mask = np.zeros((height, width), dtype=bool)
    for r, c in chosen:
        mask[int(r), int(c)] = True
    return mask


def _random_primitive_mask(
    max_h: int,
    max_w: int,
    *,
    exact_area: int | None = None,
    exact_bbox: tuple[int, int] | None = None,
    avoid_keys: set[tuple[int, int, bytes]] | None = None,
) -> np.ndarray | None:
    avoid = avoid_keys or set()
    for _ in range(SHAPE_ATTEMPTS):
        if exact_bbox is None:
            h = random.randint(2, max_h)
            w = random.randint(2, max_w)
        else:
            h, w = map(int, exact_bbox)
        if exact_area is None:
            lo = 4
            hi = min(int(h * w - 1), 9)
            if lo > hi:
                continue
            area = random.randint(lo, hi)
        else:
            area = int(exact_area)
            if area < 4 or area >= h * w:
                continue
        full = _grow_connected_mask(h, w, area)
        if full is None:
            continue
        mask = _crop_mask(full)
        if mask is None:
            continue
        if exact_bbox is not None and mask.shape != tuple(map(int, exact_bbox)):
            continue
        if not _interesting(mask) or _has_hole(mask) or not _is_connected(mask) or not _is_primitive(mask):
            continue
        key = _canonical_shape_key(mask)
        if key in avoid:
            continue
        return mask
    return None


def _render_candidate(mask: np.ndarray, color: int, scale: int, transform_code: int) -> tuple[np.ndarray, np.ndarray]:
    oriented = _dihedral(mask, transform_code)
    scaled = np.kron(oriented.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)
    patch = np.full(scaled.shape, BG, dtype=int)
    patch[scaled] = int(color)
    return patch, scaled


def _dilate8(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(iterations)):
        cur = out.copy()
        out[:-1, :] |= cur[1:, :]
        out[1:, :] |= cur[:-1, :]
        out[:, :-1] |= cur[:, 1:]
        out[:, 1:] |= cur[:, :-1]
        out[:-1, :-1] |= cur[1:, 1:]
        out[:-1, 1:] |= cur[1:, :-1]
        out[1:, :-1] |= cur[:-1, 1:]
        out[1:, 1:] |= cur[:-1, :-1]
    return out


def _place_patches(canvas_h: int, canvas_w: int, entries: list[dict]) -> np.ndarray | None:
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    forbidden = np.zeros((canvas_h, canvas_w), dtype=bool)
    order = sorted(entries, key=lambda entry: -int(entry["mask"].sum()))

    for entry in order:
        patch = np.asarray(entry["patch"], dtype=int)
        mask = np.asarray(entry["mask"], dtype=bool)
        h, w = patch.shape
        if h > canvas_h - 2 or w > canvas_w:
            return None
        positions = [
            (r0, c0)
            for r0 in range(1, canvas_h - h)
            for c0 in range(0, canvas_w - w + 1)
        ]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions[:PLACE_ATTEMPTS]:
            if np.any(forbidden[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            sub = canvas[r0 : r0 + h, c0 : c0 + w]
            if np.any((sub != BG) & mask):
                continue
            sub[mask] = patch[mask]
            footprint = np.zeros_like(canvas, dtype=bool)
            footprint[r0 : r0 + h, c0 : c0 + w] = mask
            forbidden |= _dilate8(footprint, iterations=1)
            placed = True
            entry["placed_at"] = (int(r0), int(c0))
            break
        if not placed:
            return None
    return canvas


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    assert h <= 30 and w <= 30

    top_h = 0
    while top_h < h and np.any(g[top_h] != bg):
        top_h += 1
    assert 0 < top_h < h, "top panel band must be followed by at least one bg row"

    top = g[:top_h]
    sep_candidates = []
    for color in sorted(set(map(int, np.unique(top))) - {bg}):
        coords = np.argwhere(top == int(color))
        if coords.shape[0] != top_h:
            continue
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if rows.size == top_h and cols.size == 1:
            sep_candidates.append((int(color), int(cols[0])))
    assert len(sep_candidates) == 1, "separator line must be unique"
    sep_color, sep_c = sep_candidates[0]

    left = top[:, :sep_c]
    right = top[:, sep_c + 1 :]
    assert left.shape == right.shape and left.size > 0

    left_mask = left != bg
    right_mask = right != bg
    diff = left_mask ^ right_mask
    diff_box = bbox(diff)
    assert diff_box is not None
    diff_mask = crop_bbox(diff.astype(np.uint8), diff_box).astype(bool)
    diff_key = _canonical_shape_key(diff_mask)

    bottom = g[top_h + 1 :, :]
    candidates = []
    matches = []
    for comp in find_components(bottom, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(bottom, (r0, c0, r1, c1))
        color = int(comp["color"])
        mask = patch == color
        factor, base = _infer_block_shape(mask)
        key = _canonical_shape_key(base)
        candidate = {
            "color": int(color),
            "patch": patch,
            "mask": mask,
            "factor": int(factor),
            "base": base,
            "key": key,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
        }
        if key == diff_key:
            matches.append(len(candidates))
        candidates.append(candidate)

    return {
        "bg": int(bg),
        "top_h": int(top_h),
        "sep_color": int(sep_color),
        "sep_c": int(sep_c),
        "left_mask": left_mask,
        "right_mask": right_mask,
        "diff_mask": diff_mask,
        "diff_key": diff_key,
        "candidates": candidates,
        "matches": matches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden symmetric-difference mask and return its unique scaled, reoriented candidate sprite.

    Input structure:
    - The top band contains two equal-size side-by-side panels separated by one vertical line; the two panels share one background color and each panel has exactly one foreground color.
    - Lower rows contain several isolated monochrome candidate sprites on the same background.
    - Each candidate sprite is built from a tight boolean shape enlarged by a per-sprite integer block scale.

    Rule:
    - Use panel position alignment and mark cells where exactly one panel is foreground to obtain the hidden difference shape.
    - For every lower candidate, crop it, infer its largest valid block scale, downsample it to block level, and compare that shape against the hidden difference under all rotations and flips.
    - Return the unique matching candidate in its original color, original orientation, and original enlarged size, cropped to its minimal bounding box.

    Key insight:
    - The target shape never appears directly in the panels; only the foreground symmetric difference reveals it, and only after candidate block-downsampling do the disguised orientations become comparable.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    assert len(info["matches"]) == 1, "there must be exactly one matching candidate"
    return np.asarray(info["candidates"][info["matches"][0]]["patch"], dtype=int)


def _gen_panel_pair() -> dict | None:
    for _ in range(PAIR_ATTEMPTS):
        panel_h = random.randint(*PANEL_H_RANGE)
        panel_w = random.randint(*PANEL_W_RANGE)
        max_area = min(panel_h * panel_w - 2, 15)
        min_area = max(7, panel_h)
        left = _grow_connected_mask(panel_h, panel_w, random.randint(min_area, max_area))
        right = _grow_connected_mask(panel_h, panel_w, random.randint(min_area, max_area))
        if left is None or right is None:
            continue

        overlap = left & right
        only_left = left & ~right
        only_right = right & ~left
        diff = left ^ right

        if not overlap.any() or not only_left.any() or not only_right.any():
            continue
        if not _is_connected(diff):
            continue

        diff_crop = _crop_mask(diff)
        left_crop = _crop_mask(left)
        right_crop = _crop_mask(right)
        overlap_crop = _crop_mask(overlap)
        union_crop = _crop_mask(left | right)
        left_only_crop = _crop_mask(only_left)
        right_only_crop = _crop_mask(only_right)
        assert diff_crop is not None and left_crop is not None and right_crop is not None

        if not (_interesting(diff_crop) and not _has_hole(diff_crop) and _is_primitive(diff_crop)):
            continue
        if not (DIFF_AREA_RANGE[0] <= int(diff_crop.sum()) <= DIFF_AREA_RANGE[1]):
            continue
        if max(diff_crop.shape) > MAX_BASE_DIM:
            continue

        diff_key = _canonical_shape_key(diff_crop)
        if any(
            mask is not None and _canonical_shape_key(mask) == diff_key
            for mask in (left_crop, right_crop, overlap_crop, union_crop)
        ):
            continue

        visible_sources = []
        for name, mask in (
            ("left", left_crop),
            ("right", right_crop),
            ("union", union_crop),
            ("overlap", overlap_crop),
            ("left_only", left_only_crop),
            ("right_only", right_only_crop),
        ):
            if mask is None:
                continue
            if not (_interesting(mask) and not _has_hole(mask) and _is_connected(mask) and _is_primitive(mask)):
                continue
            key = _canonical_shape_key(mask)
            if key == diff_key:
                continue
            visible_sources.append((name, mask))

        if not visible_sources:
            continue

        top = np.full((panel_h, 2 * panel_w + 1), BG, dtype=int)
        return {
            "panel_h": int(panel_h),
            "panel_w": int(panel_w),
            "left_mask": left,
            "right_mask": right,
            "diff_mask": diff_crop,
            "diff_key": diff_key,
            "visible_sources": visible_sources,
            "top_template": top,
        }
    return None


def _candidate_shapes(meta: dict) -> list[tuple[str, np.ndarray]] | None:
    diff_mask = np.asarray(meta["diff_mask"], dtype=bool)
    diff_key = _canonical_shape_key(diff_mask)
    avoid = {diff_key}
    shapes: list[tuple[str, np.ndarray]] = [("target", diff_mask)]

    visible_choices = list(meta["visible_sources"])
    random.shuffle(visible_choices)
    src_name, src_mask = visible_choices[0]
    shapes.append((src_name, np.asarray(src_mask, dtype=bool)))
    avoid.add(_canonical_shape_key(src_mask))

    same_area = _random_primitive_mask(
        min(meta["panel_h"], MAX_BASE_DIM),
        min(meta["panel_w"], MAX_BASE_DIM),
        exact_area=int(diff_mask.sum()),
        avoid_keys=avoid,
    )
    if same_area is None:
        return None
    shapes.append(("same_area", same_area))
    avoid.add(_canonical_shape_key(same_area))

    same_bbox = _random_primitive_mask(
        min(meta["panel_h"], MAX_BASE_DIM),
        min(meta["panel_w"], MAX_BASE_DIM),
        exact_bbox=(int(diff_mask.shape[0]), int(diff_mask.shape[1])),
        avoid_keys=avoid,
    )
    if same_bbox is not None:
        shapes.append(("same_bbox", same_bbox))
        avoid.add(_canonical_shape_key(same_bbox))

    desired = random.randint(*NUM_CANDIDATES_RANGE)
    while len(shapes) < desired:
        extra = _random_primitive_mask(
            min(meta["panel_h"], MAX_BASE_DIM),
            min(meta["panel_w"], MAX_BASE_DIM),
            avoid_keys=avoid,
        )
        if extra is None:
            return None
        shapes.append(("extra", extra))
        avoid.add(_canonical_shape_key(extra))

    return shapes


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    meta = _gen_panel_pair()
    if meta is None:
        return None

    candidate_shapes = _candidate_shapes(meta)
    if candidate_shapes is None:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    left_color = palette.pop()
    right_color = palette.pop()
    sep_color = palette.pop()
    if len(palette) < len(candidate_shapes):
        return None

    top = np.full((meta["panel_h"], 2 * meta["panel_w"] + 1), BG, dtype=int)
    top[:, meta["panel_w"]] = int(sep_color)
    top[:, : meta["panel_w"]][meta["left_mask"]] = int(left_color)
    top[:, meta["panel_w"] + 1 :][meta["right_mask"]] = int(right_color)

    candidate_entries = []
    target_patch = None
    scaled_non_target = 0

    for idx, (name, base_mask) in enumerate(candidate_shapes):
        color = int(palette[idx])
        if name == "target":
            scale = random.choice(TARGET_SCALE_CHOICES)
            transform_code = random.choice([1, 2, 3, 4, 5, 6, 7]) if random.random() < 0.8 else 0
        else:
            scale = random.choice(OTHER_SCALE_CHOICES)
            transform_code = random.randint(0, 7)
            if scale > 1:
                scaled_non_target += 1
        patch, patch_mask = _render_candidate(base_mask, color, scale, transform_code)
        entry = {
            "name": name,
            "base_mask": base_mask,
            "patch": patch,
            "mask": patch_mask,
            "scale": int(scale),
            "transform_code": int(transform_code),
            "key": _canonical_shape_key(base_mask),
        }
        if name == "target":
            target_patch = patch.copy()
        candidate_entries.append(entry)

    assert target_patch is not None
    if scaled_non_target == 0:
        non_targets = [entry for entry in candidate_entries if entry["name"] != "target"]
        bump = random.choice(non_targets)
        bump["scale"] = random.choice((2, 3))
        bump["patch"], bump["mask"] = _render_candidate(
            bump["base_mask"], int(np.max(bump["patch"])), bump["scale"], bump["transform_code"]
        )

    gap = random.randint(*TOP_GAP_RANGE)
    max_patch_h = max(int(entry["patch"].shape[0]) for entry in candidate_entries)
    min_bottom_h = max(11, max_patch_h + 2)
    max_bottom_h = 30 - meta["panel_h"] - gap
    if min_bottom_h > max_bottom_h:
        return None
    bottom_h = random.randint(min_bottom_h, max_bottom_h)
    bottom = _place_patches(bottom_h, top.shape[1], candidate_entries)
    if bottom is None:
        return None

    grid = np.vstack([top, np.full((gap, top.shape[1]), BG, dtype=int), bottom])
    if max(grid.shape) > 30:
        return None

    info = _analyze(grid)
    if len(info["candidates"]) != len(candidate_entries):
        return None
    if len(info["matches"]) != 1:
        return None
    if info["candidates"][info["matches"][0]]["factor"] <= 1:
        return None
    if not np.array_equal(main(grid), target_patch):
        return None

    left_mask = info["left_mask"]
    right_mask = info["right_mask"]
    overlap = left_mask & right_mask
    only_left = left_mask & ~right_mask
    only_right = right_mask & ~left_mask
    if not overlap.any() or not only_left.any() or not only_right.any():
        return None

    candidate_keys = [entry["key"] for entry in candidate_entries]
    if candidate_keys.count(meta["diff_key"]) != 1:
        return None
    visible_keys = {_canonical_shape_key(mask) for _, mask in meta["visible_sources"]}
    if not any(key in visible_keys for key in candidate_keys if key != meta["diff_key"]):
        return None
    if sum(int(entry["base_mask"].sum()) == int(meta["diff_mask"].sum()) for entry in candidate_entries) < 2:
        return None

    return grid.astype(int), target_patch.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, _target = built
        return grid
    raise RuntimeError("generate_input: failed to build a valid instance")

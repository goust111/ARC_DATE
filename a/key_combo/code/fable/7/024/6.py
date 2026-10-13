from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, label_mask, paste


GREY = Color.GREY
ALL_COLORS = tuple(range(10))
NON_GREY_COLORS = tuple(color for color in ALL_COLORS if color != GREY)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_PROTO_SIDE = 8
MIN_PROTO_SIDE = 4
MIN_COPY_COUNT = 3
MAX_COPY_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400


def _mask_points(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, coord)) for coord in np.argwhere(mask)}


def _is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    _lab, n = label_mask(mask, connectivity=4)
    return int(n) == 1


def _grow_polyomino(height: int, width: int, area: int, rng: np.random.Generator) -> np.ndarray:
    seed = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(0, len(frontier)))])

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        and mask[-1].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _prototype_ok(mask: np.ndarray) -> bool:
    if not _is_connected(mask):
        return False
    if not _touches_all_sides(mask):
        return False
    if bool(mask.all()):
        return False
    area = int(mask.sum())
    h, w = mask.shape
    if area < max(h + w - 2, 7):
        return False
    if area > h * w - 2:
        return False
    return True


def _sample_prototype(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        height = int(rng.integers(MIN_PROTO_SIDE, MAX_PROTO_SIDE + 1))
        width = int(rng.integers(MIN_PROTO_SIDE, MAX_PROTO_SIDE + 1))
        area_lo = max(height + width - 2, (height * width) // 2)
        area_hi = height * width - 2
        if area_lo > area_hi:
            continue
        area = int(rng.integers(area_lo, area_hi + 1))
        mask = _grow_polyomino(height, width, area, rng)
        if _prototype_ok(mask):
            return mask
    raise RuntimeError("failed to sample a non-degenerate prototype")


def _copy_signature(copy_info: dict) -> tuple:
    pts = tuple(sorted(copy_info["rel_points"]))
    return copy_info["dims"], copy_info["anchor_offset"], pts


def _enumerate_copy_candidates(proto_mask: np.ndarray, anchor: tuple[int, int]) -> list[dict]:
    h, w = proto_mask.shape
    ar, ac = anchor
    proto_pts = _mask_points(proto_mask)
    candidates: list[dict] = []
    seen: set[tuple] = set()
    min_area = max(4, int(proto_mask.sum()) // 2)

    for top_trim in range(ar + 1):
        for bottom_trim in range(h - ar):
            for left_trim in range(ac + 1):
                for right_trim in range(w - ac):
                    if top_trim == bottom_trim == left_trim == right_trim == 0:
                        continue

                    rows = slice(top_trim, h - bottom_trim)
                    cols = slice(left_trim, w - right_trim)
                    subset = proto_mask[rows, cols]
                    if not bool(subset.any()):
                        continue

                    local_coords = np.argwhere(subset)
                    if local_coords.shape[0] < min_area:
                        continue
                    if not _is_connected(subset):
                        continue

                    rel_anchor = (ar - top_trim, ac - left_trim)
                    if not bool(subset[rel_anchor]):
                        continue

                    full_subset = np.zeros_like(proto_mask, dtype=bool)
                    full_subset[rows, cols] = subset
                    if _mask_points(full_subset) == proto_pts:
                        continue

                    bb = bbox(subset)
                    assert bb is not None
                    tight = crop_bbox(subset, bb).astype(bool)
                    tight_coords = np.argwhere(tight)
                    bbox_r0, bbox_c0, _bbox_r1, _bbox_c1 = map(int, bb)
                    anchor_offset = (rel_anchor[0] - bbox_r0, rel_anchor[1] - bbox_c0)
                    rel_points = {
                        (int(r) - anchor_offset[0], int(c) - anchor_offset[1])
                        for r, c in tight_coords
                    }
                    candidate = {
                        "tight_mask": tight,
                        "dims": tuple(map(int, tight.shape)),
                        "anchor_offset": tuple(map(int, anchor_offset)),
                        "rel_points": rel_points,
                        "global_points": _mask_points(full_subset),
                    }
                    sig = _copy_signature(candidate)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    candidates.append(candidate)

    return candidates


def _choose_copies(
    proto_mask: np.ndarray, anchor: tuple[int, int], rng: np.random.Generator
) -> list[dict]:
    proto_pts = _mask_points(proto_mask)
    candidates = _enumerate_copy_candidates(proto_mask, anchor)
    if len(candidates) < MIN_COPY_COUNT:
        raise RuntimeError("not enough damaged copy candidates")

    order = list(rng.permutation(len(candidates)))
    for copy_count in rng.permutation(np.arange(MIN_COPY_COUNT, MAX_COPY_COUNT + 1)):
        for _ in range(2500):
            picked_idx = order[:]
            rng.shuffle(picked_idx)
            chosen = [candidates[idx] for idx in picked_idx[: int(copy_count)]]

            dims = {copy["dims"] for copy in chosen}
            if len(dims) != len(chosen):
                continue

            anchor_offsets = {copy["anchor_offset"] for copy in chosen}
            if len(anchor_offsets) < 2:
                continue

            union_pts = set().union(*(copy["global_points"] for copy in chosen))
            if union_pts != proto_pts:
                continue

            if any(copy["global_points"] == proto_pts for copy in chosen):
                continue

            necessary = True
            for i, copy in enumerate(chosen):
                others = set().union(
                    *(chosen[j]["global_points"] for j in range(len(chosen)) if j != i)
                )
                if others == proto_pts:
                    necessary = False
                    break
                if not (copy["global_points"] - others):
                    necessary = False
                    break
            if not necessary:
                continue

            return chosen

    raise RuntimeError("failed to choose a non-degenerate damaged-copy set")


def _build_copy_patch(
    tight_mask: np.ndarray, anchor_offset: tuple[int, int], body_color: int, bg: int
) -> np.ndarray:
    patch = np.full(tight_mask.shape, bg, dtype=int)
    patch[tight_mask] = int(body_color)
    patch[int(anchor_offset[0]), int(anchor_offset[1])] = GREY
    return patch


def _build_frame_patch(inner_h: int, inner_w: int, frame_color: int, bg: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    return patch


def _random_place_patches(
    patches: list[np.ndarray], bg: int, rng: np.random.Generator
) -> np.ndarray | None:
    for _ in range(150):
        height = int(rng.integers(18, 31))
        width = int(rng.integers(18, 31))
        canvas = np.full((height, width), bg, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)
        order = list(rng.permutation(len(patches)))
        placements: dict[int, tuple[int, int]] = {}
        ok = True

        for idx in order:
            patch = patches[int(idx)]
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(height - ph + 1):
                for c0 in range(width - pw + 1):
                    if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                ok = False
                break

            r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
            placements[int(idx)] = (r0, c0)
            paste(canvas, patch, r0, c0, transparent=bg)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc1 = min(width, c0 + pw + 1)
            blocked[rr0:rr1, cc0:cc1] = True

        if ok:
            return canvas

    return None


def _frame_info(patch: np.ndarray, bg: int) -> tuple[int, int, int] | None:
    colors = sorted(set(map(int, np.unique(patch))) - {int(bg)})
    if len(colors) != 1:
        return None
    frame_color = colors[0]
    h, w = patch.shape
    if h < 3 or w < 3:
        return None

    expected = np.full((h, w), bg, dtype=int)
    expected[0, :] = frame_color
    expected[-1, :] = frame_color
    expected[:, 0] = frame_color
    expected[:, -1] = frame_color
    if not np.array_equal(patch, expected):
        return None
    return frame_color, h - 2, w - 2


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    frame: dict | None = None
    copies: list[dict] = []
    for component in components:
        patch = crop_bbox(g, component["bbox"])
        colors = sorted(set(map(int, np.unique(patch))) - {bg})
        if not colors:
            continue

        if len(colors) == 1:
            info = _frame_info(patch, bg)
            if info is None:
                raise AssertionError("unexpected monochrome non-frame component")
            frame_color, inner_h, inner_w = info
            if frame is not None:
                raise AssertionError("expected exactly one frame")
            frame = {
                "bbox": tuple(map(int, component["bbox"])),
                "color": int(frame_color),
                "inner_shape": (int(inner_h), int(inner_w)),
            }
            continue

        if len(colors) != 2 or GREY not in colors or int(np.count_nonzero(patch == GREY)) != 1:
            raise AssertionError("unexpected multicolor component")
        body_color = next(color for color in colors if color != GREY)
        anchor = tuple(map(int, np.argwhere(patch == GREY)[0]))
        mask = patch != bg
        mask[int(anchor[0]), int(anchor[1])] = True
        copies.append(
            {
                "body_color": int(body_color),
                "bbox": tuple(map(int, component["bbox"])),
                "mask": mask.astype(bool),
                "dims": tuple(map(int, mask.shape)),
                "anchor": anchor,
            }
        )

    assert frame is not None, "frame not found"
    assert len(copies) >= 2, "damaged copies not found"

    body_colors = {copy["body_color"] for copy in copies}
    assert len(body_colors) == 1, "copies must share one body color"
    body_color = body_colors.pop()
    assert body_color != frame["color"], "frame color must differ from body color"

    return {
        "bg": bg,
        "body_color": int(body_color),
        "frame": frame,
        "copies": copies,
    }


def _reconstruct_prototype(copies: list[dict]) -> tuple[np.ndarray, tuple[int, int]]:
    rel_points: set[tuple[int, int]] = set()
    for copy in copies:
        ar, ac = copy["anchor"]
        for r, c in np.argwhere(copy["mask"]):
            rel_points.add((int(r) - int(ar), int(c) - int(ac)))

    min_r = min(r for r, _c in rel_points)
    min_c = min(c for _r, c in rel_points)
    max_r = max(r for r, _c in rel_points)
    max_c = max(c for _r, c in rel_points)

    proto = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for r, c in rel_points:
        proto[r - min_r, c - min_c] = True
    anchor = (-min_r, -min_c)
    return proto, anchor


def _scene_semantics(scene: dict) -> dict:
    copies = scene["copies"]
    proto_mask, proto_anchor = _reconstruct_prototype(copies)
    proto_pts = _mask_points(proto_mask)

    dims = [copy["dims"] for copy in copies]
    assert len(set(dims)) == len(dims), "copy bbox sizes must be pairwise distinct"

    anchor_offsets = {copy["anchor"] for copy in copies}
    assert len(anchor_offsets) >= 2, "anchor positions must vary across copy bboxes"

    aligned_sets: list[set[tuple[int, int]]] = []
    for copy in copies:
        rel = {
            (int(r) - int(copy["anchor"][0]), int(c) - int(copy["anchor"][1]))
            for r, c in np.argwhere(copy["mask"])
        }
        aligned = {(r + proto_anchor[0], c + proto_anchor[1]) for r, c in rel}
        aligned_sets.append(aligned)
        assert aligned < proto_pts, "no damaged copy may already be complete"

    for i, aligned in enumerate(aligned_sets):
        others = set().union(*(aligned_sets[j] for j in range(len(aligned_sets)) if j != i))
        assert others != proto_pts, "every copy must matter to the final union"
        assert aligned - others, "every copy must contribute a unique pixel"

    assert set().union(*aligned_sets) == proto_pts, "copies must union to the full prototype"

    frame = scene["frame"]
    assert frame["inner_shape"] == proto_mask.shape, "frame interior must match prototype bbox"

    return {
        "proto_mask": proto_mask,
        "proto_anchor": proto_anchor,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align damaged copies by their gray anchors, union them into the hidden prototype, and print it inside the empty frame.

    Input structure:
    - A dominant background contains several separated damaged copies plus one hollow rectangular frame.
    - Each damaged copy uses one body color and exactly one gray anchor pixel; the body pixels are an incomplete subset of the same hidden prototype.
    - The gray anchor marks the same prototype cell in every copy, while copy bounding boxes differ because different boundary pixels are missing.

    Rule:
    - Treat the gray anchor cell as part of the body color.
    - Crop each damaged copy to its own bbox, align all copies so their gray anchors coincide, and take the union of their body pixels to recover the complete prototype.
    - Draw that recovered prototype inside the frame interior, using the frame interior as the prototype bbox canvas, while leaving all original copies and the frame itself unchanged.

    Key insight:
    - No complete prototype ever appears directly; only anchor-based registration can assemble it, because bbox-based alignment is unreliable when each damaged copy is cropped differently.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    info = _scene_semantics(scene)

    proto_mask = info["proto_mask"]
    frame = scene["frame"]
    bg = scene["bg"]
    body_color = scene["body_color"]

    out = np.array(grid, dtype=int, copy=True)
    r0, c0, _r1, _c1 = frame["bbox"]
    inner_r0 = int(r0) + 1
    inner_c0 = int(c0) + 1

    patch = np.full(proto_mask.shape, bg, dtype=int)
    patch[proto_mask] = int(body_color)
    paste(out, patch, inner_r0, inner_c0, transparent=bg)
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        proto_mask = _sample_prototype(rng)
        anchor_candidates = [tuple(map(int, coord)) for coord in np.argwhere(proto_mask)]
        rng.shuffle(anchor_candidates)

        copies: list[dict] | None = None
        anchor: tuple[int, int] | None = None
        for candidate_anchor in anchor_candidates:
            try:
                candidate_copies = _choose_copies(proto_mask, candidate_anchor, rng)
            except RuntimeError:
                continue
            copies = candidate_copies
            anchor = candidate_anchor
            break

        if copies is None or anchor is None:
            continue

        palette = list(rng.choice(NON_GREY_COLORS, size=3, replace=False))
        bg, body_color, frame_color = map(int, palette)

        copy_patches = [
            _build_copy_patch(copy["tight_mask"], copy["anchor_offset"], body_color, bg)
            for copy in copies
        ]
        frame_patch = _build_frame_patch(proto_mask.shape[0], proto_mask.shape[1], frame_color, bg)
        patches = copy_patches + [frame_patch]

        grid = _random_place_patches(patches, bg, rng)
        if grid is None:
            continue

        expected = grid.copy()
        frame_scene = _extract_scene(grid)["frame"]
        inner_patch = np.full(proto_mask.shape, bg, dtype=int)
        inner_patch[proto_mask] = body_color
        paste(expected, inner_patch, int(frame_scene["bbox"][0]) + 1, int(frame_scene["bbox"][1]) + 1, transparent=bg)

        try:
            scene = _extract_scene(grid)
            _scene_semantics(scene)
        except Exception:
            continue

        if np.array_equal(main(grid), expected):
            return grid

    raise RuntimeError("failed to generate a valid residual-overlay ARC input")

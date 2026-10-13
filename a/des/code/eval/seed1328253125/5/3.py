from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(bool).tobytes()


def _transform(mask: np.ndarray, code: int) -> np.ndarray:
    base = np.asarray(mask, dtype=bool)
    if code < 4:
        return np.rot90(base, code)
    return np.rot90(np.fliplr(base), code - 4)


def _distinct_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for code in range(8):
        variant = _transform(mask, code)
        sig = _mask_signature(variant)
        if sig not in seen:
            seen.add(sig)
            out.append(variant)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_signature(variant) for variant in _distinct_orientations(mask))


def _matches_under_dihedral(a: np.ndarray, b: np.ndarray) -> bool:
    return any(np.array_equal(a, variant) for variant in _distinct_orientations(b))


def _reflection_only_codes(mask: np.ndarray) -> list[int]:
    rotation_sigs = {_mask_signature(_transform(mask, code)) for code in range(4)}
    return [
        code
        for code in range(4, 8)
        if _mask_signature(_transform(mask, code)) not in rotation_sigs
    ]


def _border_is_color(subgrid: np.ndarray, color: int) -> bool:
    return bool(
        np.all(subgrid[0, :] == color)
        and np.all(subgrid[-1, :] == color)
        and np.all(subgrid[:, 0] == color)
        and np.all(subgrid[:, -1] == color)
    )


def _hole_descriptors(subgrid: np.ndarray, bg: int) -> list[dict]:
    holes: list[dict] = []
    for coords in _components(subgrid == bg):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == 0 or c0 == 0 or r1 == subgrid.shape[0] - 1 or c1 == subgrid.shape[1] - 1:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "mask": mask,
                "bbox": (r0, c0, r1, c1),
                "coords": coords.copy(),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill same-color template holes using scattered fragments that match by shape
    under rotations and reflections, then erase the original fragments.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for color in sorted(set(map(int, g.ravel())) - {bg}):
        comps = _components(g == color)
        assert len(comps) >= 2, "each template color must have one template and at least one fragment"

        template_candidates: list[dict] = []
        for idx, coords in enumerate(comps):
            r0, c0, r1, c1 = _bbox(coords)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if not _border_is_color(sub, color):
                continue
            holes = _hole_descriptors(sub, bg)
            if holes:
                template_candidates.append(
                    {
                        "index": idx,
                        "bbox": (r0, c0, r1, c1),
                        "holes": holes,
                        "area": int(coords.shape[0]),
                    }
                )

        assert template_candidates, "template component not found"
        template = max(template_candidates, key=lambda item: item["area"])
        assert sum(1 for cand in template_candidates if cand["area"] == template["area"]) == 1

        hole_masks = [hole["mask"] for hole in template["holes"]]
        fragment_coords = [coords for idx, coords in enumerate(comps) if idx != template["index"]]
        fragment_masks = [_mask_from_coords(coords) for coords in fragment_coords]
        assert len(fragment_masks) == len(hole_masks), "hole and fragment counts must match"

        used = [False] * len(fragment_masks)
        for hole_mask in sorted(hole_masks, key=lambda item: (-int(item.sum()), item.shape)):
            match_idx = None
            for idx, frag_mask in enumerate(fragment_masks):
                if used[idx]:
                    continue
                if _matches_under_dihedral(hole_mask, frag_mask):
                    match_idx = idx
                    break
            assert match_idx is not None, "every hole must have a matching fragment"
            used[match_idx] = True

        for coords in fragment_coords:
            out[coords[:, 0], coords[:, 1]] = bg
        tr0, tc0, _, _ = template["bbox"]
        for hole in template["holes"]:
            abs_coords = hole["coords"] + np.array([tr0, tc0], dtype=int)
            out[abs_coords[:, 0], abs_coords[:, 1]] = color

    return out


# --------------- generator ---------------

MAX_GEN_ATTEMPTS = 400
FRAME_BUILD_ATTEMPTS = 80
FRAME_PLACEMENT_ATTEMPTS = 120
FRAGMENT_PLACEMENT_ATTEMPTS = 160

TEMPLATE_COUNT_OPTIONS = (1, 2, 3)
FRAME_H_RANGE = (6, 10)
FRAME_W_RANGE = (6, 11)


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 1], [1, 0]],
        [[1, 1], [0, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1, 1]],
        [[1, 0, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [0, 1, 1]],
        [[1, 1, 0], [1, 1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            out.append(mask)
    return out


SHAPE_LIBRARY = _build_shape_library()


def _sample_canvas_shape(template_count: int) -> tuple[int, int]:
    low = 17 + 4 * (template_count - 1)
    high = min(30, low + 6)
    return random.randint(low, high), random.randint(low, high)


def _bbox_is_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    return not bool(np.any(occ[rr0:rr1, cc0:cc1]))


def _mark_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _pick_location(
    occ: np.ndarray, patch_shape: tuple[int, int], attempts: int
) -> tuple[int, int] | None:
    h, w = patch_shape
    grid_h, grid_w = occ.shape
    if h + 2 > grid_h or w + 2 > grid_w:
        return None
    for _ in range(attempts):
        r0 = random.randint(1, grid_h - h - 1)
        c0 = random.randint(1, grid_w - w - 1)
        if _bbox_is_free(occ, r0, c0, h, w, pad=1):
            return r0, c0
    return None


def _max_holes_for_frame(frame_h: int, frame_w: int) -> int:
    if frame_h >= 9 and frame_w >= 10:
        return 3
    if frame_h >= 7 and frame_w >= 8:
        return 2
    return 1


def _build_frame(frame_h: int, frame_w: int) -> tuple[np.ndarray, list[dict]] | None:
    for _ in range(FRAME_BUILD_ATTEMPTS):
        target_holes = random.randint(1, _max_holes_for_frame(frame_h, frame_w))
        template = np.ones((frame_h, frame_w), dtype=bool)
        occ = np.zeros((frame_h, frame_w), dtype=bool)
        used_keys: set[tuple[tuple[int, int], bytes]] = set()
        holes: list[dict] = []

        candidates = SHAPE_LIBRARY[:]
        random.shuffle(candidates)
        success = True
        for _hole_idx in range(target_holes):
            placed = False
            random.shuffle(candidates)
            for shape in candidates:
                key = _canonical_key(shape)
                if key in used_keys:
                    continue
                variants = _distinct_orientations(shape)
                random.shuffle(variants)
                for variant in variants:
                    sh, sw = map(int, variant.shape)
                    if sh > frame_h - 2 or sw > frame_w - 2:
                        continue
                    for _ in range(40):
                        r0 = random.randint(1, frame_h - sh - 1)
                        c0 = random.randint(1, frame_w - sw - 1)
                        if not _bbox_is_free(occ, r0, c0, sh, sw, pad=1):
                            continue
                        _mark_bbox(occ, r0, c0, sh, sw, pad=1)
                        sub = template[r0 : r0 + sh, c0 : c0 + sw]
                        sub[variant] = False
                        holes.append(
                            {
                                "mask": variant.copy(),
                                "coords": np.argwhere(variant) + np.array([r0, c0], dtype=int),
                            }
                        )
                        used_keys.add(key)
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                success = False
                break

        if success and holes:
            return template, holes
    return None


def _choose_fragment_transforms(hole_masks: list[np.ndarray]) -> list[int]:
    codes = [random.randrange(8) for _ in hole_masks]
    nontrivial = [
        idx
        for idx, mask in enumerate(hole_masks)
        if any(not np.array_equal(_transform(mask, code), mask) for code in range(1, 8))
    ]
    produced = [_transform(mask, code) for mask, code in zip(hole_masks, codes)]
    if nontrivial and all(np.array_equal(prod, mask) for prod, mask in zip(produced, hole_masks)):
        choice = random.choice(nontrivial)
        alternatives = [
            code
            for code in range(1, 8)
            if not np.array_equal(_transform(hole_masks[choice], code), hole_masks[choice])
        ]
        codes[choice] = random.choice(alternatives)

    if hole_masks and random.random() < 0.5:
        reflection_choices = [
            (idx, _reflection_only_codes(mask))
            for idx, mask in enumerate(hole_masks)
        ]
        reflection_choices = [(idx, opts) for idx, opts in reflection_choices if opts]
        if reflection_choices:
            chosen_idx, opts = random.choice(reflection_choices)
            codes[chosen_idx] = random.choice(opts)
    return codes


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    template_count = random.choice(TEMPLATE_COUNT_OPTIONS)
    grid_h, grid_w = _sample_canvas_shape(template_count)

    color_pool = list(range(10))
    random.shuffle(color_pool)
    bg = color_pool[0]
    template_colors = color_pool[1 : template_count + 1]

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    all_holes: list[dict] = []

    for color in template_colors:
        placed_frame = False
        for _ in range(FRAME_PLACEMENT_ATTEMPTS):
            frame_h = random.randint(*FRAME_H_RANGE)
            frame_w = random.randint(*FRAME_W_RANGE)
            frame = _build_frame(frame_h, frame_w)
            if frame is None:
                continue
            template_mask, holes = frame
            pos = _pick_location(occ, (frame_h, frame_w), attempts=20)
            if pos is None:
                continue
            r0, c0 = pos
            _mark_bbox(occ, r0, c0, frame_h, frame_w, pad=1)

            frame_view = inp[r0 : r0 + frame_h, c0 : c0 + frame_w]
            frame_view[template_mask] = color
            out[r0 : r0 + frame_h, c0 : c0 + frame_w] = color

            for hole in holes:
                all_holes.append(
                    {
                        "color": color,
                        "mask": hole["mask"],
                        "coords": hole["coords"] + np.array([r0, c0], dtype=int),
                    }
                )
            placed_frame = True
            break
        if not placed_frame:
            return None

    transform_codes = _choose_fragment_transforms([hole["mask"] for hole in all_holes])
    for hole, code in zip(all_holes, transform_codes):
        fragment_mask = _transform(hole["mask"], code)
        fh, fw = map(int, fragment_mask.shape)
        pos = _pick_location(occ, (fh, fw), attempts=FRAGMENT_PLACEMENT_ATTEMPTS)
        if pos is None:
            return None
        r0, c0 = pos
        _mark_bbox(occ, r0, c0, fh, fw, pad=1)
        patch = inp[r0 : r0 + fh, c0 : c0 + fw]
        patch[fragment_mask] = hole["color"]

    counts = Counter(map(int, inp.ravel()))
    bg_count = counts.pop(bg)
    if not counts or bg_count <= max(counts.values()):
        return None

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")

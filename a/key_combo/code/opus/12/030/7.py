from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
REFERENCE_COLORS = (1, 2, 3, 8, 9)
OUTSIDE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
MIN_GRID = 22
MAX_GRID = 29


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_cells(
    grid: np.ndarray,
    allowed_colors: set[int],
) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in allowed_colors:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            components.append({"color": color, "coords": coords, "bbox": _bbox_from_coords(coords)})

    return components


def _shape_signature(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox_from_coords(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _mask_from_signature(sig: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in sig) + 1
    w = max(c for _, c in sig) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in sig:
        mask[r, c] = True
    return mask


def _has_axis_symmetry(sig: tuple[tuple[int, int], ...]) -> bool:
    mask = _mask_from_signature(sig)
    return bool(np.array_equal(mask, mask[::-1, :]) or np.array_equal(mask, mask[:, ::-1]))


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _component_cells(grid, {GREY}):
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        border_count = 2 * (h + w) - 4
        if len(coords) != border_count:
            continue
        border_ok = True
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border != (int(grid[rr, cc]) == GREY):
                    border_ok = False
                    break
            if not border_ok:
                break
        if border_ok:
            frames.append({"bbox": (r0, c0, r1, c1), "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1)})
    return frames


def _inside_bbox(bbox: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    or0, oc0, or1, oc1 = outer
    return or0 <= r0 and oc0 <= c0 and r1 <= or1 and c1 <= oc1


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    frames = _find_frames(grid)
    sprite_components = _component_cells(grid, set(range(10)) - {BLACK, GREY})

    references: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}
    inside_ids: set[int] = set()

    for idx, frame in enumerate(frames):
        interior = frame["interior"]
        inside = [comp for comp in sprite_components if _inside_bbox(comp["bbox"], interior)]
        assert len(inside) == 1, "each grey frame must contain exactly one sprite"
        comp = inside[0]
        inside_ids.add(id(comp))
        sig = _shape_signature(comp["coords"])
        assert sig not in references, "boxed reference shapes must be unique to keep the mapping unambiguous"
        references[sig] = {"color": int(comp["color"]), "frame_index": idx, "symmetric": _has_axis_symmetry(sig)}

    outside = [comp for comp in sprite_components if id(comp) not in inside_ids]
    return {"frames": frames, "references": references, "outside": outside}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only symmetry-qualified outside copies to the color of the same-shaped sprite boxed in gray.

    Input structure:
    - A black background contains 2-3 gray hollow rectangles and several free monochromatic sprites.
    - Each gray rectangle encloses exactly one non-gray monochromatic sprite.
    - Free sprites outside the boxes may match a boxed sprite's shape exactly, or may be unrelated distractors.

    Rule:
    - Detect the gray hollow rectangles and extract the enclosed reference sprites as shape-to-color examples.
    - For each free outside sprite, compare its normalized shape mask against the boxed reference shapes using exact translation-only matching.
    - If an outside sprite matches a boxed shape and its own mask has horizontal or vertical mirror symmetry, recolor that whole sprite to the boxed reference color.
    - Leave gray frames, boxed sprites, non-matching sprites, and matching-but-non-symmetric sprites unchanged.

    Key insight:
    - Shape matching alone is insufficient: color transfer is gated by whether the outside copy itself has at least one reflection symmetry axis.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_grid(g)
    references: dict[tuple[tuple[int, int], ...], dict[str, object]] = analysis["references"]
    out = g.copy()

    for comp in analysis["outside"]:
        sig = _shape_signature(comp["coords"])
        ref = references.get(sig)
        if ref is None or not _has_axis_symmetry(sig):
            continue
        for r, c in comp["coords"]:
            out[r, c] = int(ref["color"])

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set()
    frontier.update((nr, nc) for nr, nc in ((start[0] - 1, start[1]), (start[0] + 1, start[1]), (start[0], start[1] - 1), (start[0], start[1] + 1)) if 0 <= nr < h and 0 <= nc < w)

    while int(mask.sum()) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        r, c = cell
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    if int(mask.sum()) != area:
        raise ValueError("failed to grow connected mask")
    return _trim_mask(mask)


def _random_symmetric_mask() -> np.ndarray:
    for _ in range(300):
        axis = random.choice(("horizontal", "vertical"))
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if axis == "horizontal":
            half_h = (h + 1) // 2
            half = _random_connected_mask(half_h, w, random.randint(2, min(5, half_h * w)))
            hh, ww = half.shape
            full = np.zeros((hh * 2 - 1 if hh > 1 else 1, ww), dtype=bool)
            for r, c in np.argwhere(half):
                rr = int(r)
                cc = int(c)
                full[rr, cc] = True
                full[full.shape[0] - 1 - rr, cc] = True
        else:
            half_w = (w + 1) // 2
            half = _random_connected_mask(h, half_w, random.randint(2, min(5, h * half_w)))
            hh, ww = half.shape
            full = np.zeros((hh, ww * 2 - 1 if ww > 1 else 1), dtype=bool)
            for r, c in np.argwhere(half):
                rr = int(r)
                cc = int(c)
                full[rr, cc] = True
                full[rr, full.shape[1] - 1 - cc] = True

        full = _trim_mask(full)
        area = int(full.sum())
        if 3 <= area <= 8 and _is_connected(full) and _has_axis_symmetry(_shape_signature([tuple(map(int, rc)) for rc in np.argwhere(full)])):
            if area < full.shape[0] * full.shape[1]:
                return full
    raise RuntimeError("failed to sample symmetric mask")


def _random_asymmetric_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(8, h * w - 1))
        mask = _random_connected_mask(h, w, area)
        sig = _shape_signature([tuple(map(int, rc)) for rc in np.argwhere(mask)])
        if not _has_axis_symmetry(sig) and mask.shape[0] >= 2 and mask.shape[1] >= 2:
            return mask
    raise RuntimeError("failed to sample asymmetric mask")


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _mask_to_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _find_free_rect(occupied: np.ndarray, h: int, w: int, margin: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        r0 = max(0, r - margin)
        c0 = max(0, c - margin)
        r1 = min(H, r + h + margin)
        c1 = min(W, c + w + margin)
        if not bool(occupied[r0:r1, c0:c1].any()):
            return r, c
    return None


def _occupy_rect(occupied: np.ndarray, r: int, c: int, h: int, w: int, margin: int) -> None:
    H, W = occupied.shape
    r0 = max(0, r - margin)
    c0 = max(0, c - margin)
    r1 = min(H, r + h + margin)
    c1 = min(W, c + w + margin)
    occupied[r0:r1, c0:c1] = True


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = int(color)


def _draw_box(grid: np.ndarray, top: int, left: int, interior_h: int, interior_w: int) -> None:
    grid[top : top + interior_h + 2, left] = GREY
    grid[top : top + interior_h + 2, left + interior_w + 1] = GREY
    grid[top, left : left + interior_w + 2] = GREY
    grid[top + interior_h + 1, left : left + interior_w + 2] = GREY


def _sample_unique_shape(existing: set[tuple[tuple[int, int], ...]], symmetric: bool) -> np.ndarray:
    for _ in range(300):
        mask = _random_symmetric_mask() if symmetric else _random_asymmetric_mask()
        sig = _mask_signature(mask)
        if sig not in existing:
            return mask
    raise RuntimeError("failed to sample unique shape")


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(MIN_GRID, MAX_GRID)
    W = random.randint(MIN_GRID, MAX_GRID)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    ref_count = random.randint(2, 3)
    kinds = [True, False]
    while len(kinds) < ref_count:
        kinds.append(random.choice((True, False)))
    random.shuffle(kinds)

    ref_colors = random.sample(list(REFERENCE_COLORS), ref_count)
    used_sigs: set[tuple[tuple[int, int], ...]] = set()
    reference_specs: list[dict[str, object]] = []

    for idx in range(ref_count):
        symmetric = kinds[idx]
        mask = _sample_unique_shape(used_sigs, symmetric=symmetric)
        sig = _mask_signature(mask)
        used_sigs.add(sig)
        mh, mw = mask.shape
        interior_h = random.randint(max(4, mh), 6)
        interior_w = random.randint(max(4, mw), 6)
        box_h = interior_h + 2
        box_w = interior_w + 2
        loc = _find_free_rect(occupied, box_h, box_w, margin=1)
        if loc is None:
            return None
        top, left = loc
        _draw_box(grid, top, left, interior_h, interior_w)
        _occupy_rect(occupied, top, left, box_h, box_w, margin=1)

        inner_r = top + 1 + random.randint(0, interior_h - mh)
        inner_c = left + 1 + random.randint(0, interior_w - mw)
        _place_mask(grid, mask, inner_r, inner_c, ref_colors[idx])
        reference_specs.append({"mask": mask, "sig": sig, "color": ref_colors[idx], "symmetric": symmetric})

    outside_colors = list(OUTSIDE_COLORS)
    for ref in reference_specs:
        copies = 1 if ref["symmetric"] else random.randint(1, 2)
        for _ in range(copies):
            source_color = random.choice([c for c in outside_colors if c != ref["color"]])
            sprite = _mask_to_sprite(ref["mask"], source_color)
            sh, sw = sprite.shape
            loc = _find_free_rect(occupied, sh, sw, margin=1)
            if loc is None:
                return None
            top, left = loc
            _place_mask(grid, ref["mask"], top, left, source_color)
            _occupy_rect(occupied, top, left, sh, sw, margin=1)

    unmatched_sym = _sample_unique_shape(used_sigs, symmetric=True)
    used_sigs.add(_mask_signature(unmatched_sym))
    for unmatched_mask, want_sym in [(unmatched_sym, True), (_sample_unique_shape(used_sigs, symmetric=False), False)]:
        sig = _mask_signature(unmatched_mask)
        used_sigs.add(sig)
        color = random.choice(outside_colors)
        sh, sw = unmatched_mask.shape
        loc = _find_free_rect(occupied, sh, sw, margin=1)
        if loc is None:
            return None
        top, left = loc
        _place_mask(grid, unmatched_mask, top, left, color)
        _occupy_rect(occupied, top, left, sh, sw, margin=1)

    if random.random() < 0.6:
        extra_sym = random.choice((True, False))
        extra_mask = _sample_unique_shape(used_sigs, symmetric=extra_sym)
        sig = _mask_signature(extra_mask)
        color = random.choice(outside_colors)
        sh, sw = extra_mask.shape
        loc = _find_free_rect(occupied, sh, sw, margin=1)
        if loc is not None:
            top, left = loc
            _place_mask(grid, extra_mask, top, left, color)
            _occupy_rect(occupied, top, left, sh, sw, margin=1)

    if _validate_generated_grid(grid):
        return grid
    return None


def _validate_generated_grid(grid: np.ndarray) -> bool:
    analysis = _analyze_grid(grid)
    refs: dict[tuple[tuple[int, int], ...], dict[str, object]] = analysis["references"]
    outside: list[dict[str, object]] = analysis["outside"]
    if not (2 <= len(refs) <= 3):
        return False
    if not any(bool(ref["symmetric"]) for ref in refs.values()):
        return False
    if not any(not bool(ref["symmetric"]) for ref in refs.values()):
        return False

    per_ref_counts = {sig: 0 for sig in refs}
    matched_sym = 0
    matched_asym = 0
    unmatched_sym = 0
    out = main(grid)

    for comp in outside:
        sig = _shape_signature(comp["coords"])
        in_color = int(comp["color"])
        out_colors = {int(out[r, c]) for r, c in comp["coords"]}
        if len(out_colors) != 1:
            return False
        out_color = next(iter(out_colors))

        if sig in refs:
            per_ref_counts[sig] += 1
            if bool(refs[sig]["symmetric"]):
                if out_color != int(refs[sig]["color"]) or out_color == in_color:
                    return False
                matched_sym += 1
            else:
                if out_color != in_color:
                    return False
                matched_asym += 1
        else:
            if _has_axis_symmetry(sig):
                unmatched_sym += 1
            if out_color != in_color:
                return False

    if min(per_ref_counts.values()) < 1:
        return False
    if matched_sym < 1 or matched_asym < 1 or unmatched_sym < 1:
        return False

    changed = int(np.sum(grid != out))
    return changed > 0


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-gated shape-matching puzzle")

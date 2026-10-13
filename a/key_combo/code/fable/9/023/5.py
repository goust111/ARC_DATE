from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
YELLOW = 4
PINK = 6

_INV_TRANSFORM = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5, 7: 7}
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIR4]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _transform_coords(coords: np.ndarray, idx: int) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    r = arr[:, 0]
    c = arr[:, 1]
    if idx == 0:
        out = np.stack([r, c], axis=1)
    elif idx == 1:
        out = np.stack([r, -c], axis=1)
    elif idx == 2:
        out = np.stack([-r, c], axis=1)
    elif idx == 3:
        out = np.stack([-r, -c], axis=1)
    elif idx == 4:
        out = np.stack([c, r], axis=1)
    elif idx == 5:
        out = np.stack([c, -r], axis=1)
    elif idx == 6:
        out = np.stack([-c, r], axis=1)
    elif idx == 7:
        out = np.stack([-c, -r], axis=1)
    else:
        raise ValueError(f"bad transform index: {idx}")
    return out.astype(int)


def _transform_point(point: tuple[int, int], idx: int) -> tuple[int, int]:
    out = _transform_coords(np.asarray([point], dtype=int), idx)[0]
    return int(out[0]), int(out[1])


def _inverse_transform_point(point: tuple[int, int], idx: int) -> tuple[int, int]:
    return _transform_point(point, _INV_TRANSFORM[idx])


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    arr = np.asarray(coords, dtype=int)
    mins = arr.min(axis=0)
    shifted = arr - mins
    key = tuple(sorted((int(r), int(c)) for r, c in shifted))
    return key, shifted.astype(int), mins.astype(int)


def _bbox_of(cells: set[tuple[int, int]] | list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(list(cells), dtype=int)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 0].max()), int(arr[:, 1].max())


def _shift_cells(cells: set[tuple[int, int]] | list[tuple[int, int]], delta: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in cells}


def _all_transform_keys(cells: set[tuple[int, int]]) -> dict[int, tuple[tuple[int, int], ...]]:
    arr = np.asarray(sorted(cells), dtype=int)
    keys: dict[int, tuple[tuple[int, int], ...]] = {}
    for idx in range(8):
        key, _shifted, _mins = _normalize_coords(_transform_coords(arr, idx))
        keys[idx] = key
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the hidden yellow template by aligning incomplete transformed copies with their intact purple signature.

    Input structure:
    - A black background contains several separated foreground copies of the same hidden template.
    - In every copy, the purple signature cells are complete, while the yellow body cells are only partially present.
    - Different copies may use different rotations or mirrorings, and no copy is complete by itself.

    Rule:
    - Treat each separated foreground object as one damaged copy and read its purple cells as the orientation signature.
    - Use the asymmetric purple signature to map every copy back to one common coordinate frame, then take the union of all yellow cells in that frame to recover the full template body.
    - Transform that recovered yellow body back into each copy's original pose and paint any missing yellow cells, leaving existing purple and yellow cells unchanged.

    Key insight:
    - The complete template never appears directly; only the union of yellow fragments from multiple signature-aligned copies reveals it, and the purple signature is the only reliable pose key.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g != bg)
    assert len(comps) >= 2, "expected multiple damaged copies"

    per_copy: list[dict[str, object]] = []
    canonical_signature_key: tuple[tuple[int, int], ...] | None = None

    for comp in comps:
        pixels = {(int(r), int(c)) for r, c in comp}
        pink = np.asarray(sorted((r, c) for r, c in pixels if int(g[r, c]) == PINK), dtype=int)
        yellow = np.asarray(sorted((r, c) for r, c in pixels if int(g[r, c]) == YELLOW), dtype=int)
        assert len(pink) > 0 and len(yellow) > 0, "each copy must contain both signature and body remnants"

        best_key = min(_all_transform_keys({(int(r), int(c)) for r, c in pink}).values())
        if canonical_signature_key is None:
            canonical_signature_key = best_key
        assert best_key == canonical_signature_key, "all copies must share the same canonical signature"
        per_copy.append({"pink": pink, "yellow": yellow})

    assert canonical_signature_key is not None
    canonical_signature = np.asarray(canonical_signature_key, dtype=int)
    variant_keys: dict[int, tuple[tuple[int, int], ...]] = {}
    variant_mins: dict[int, np.ndarray] = {}
    for idx in range(8):
        key, _shifted, mins = _normalize_coords(_transform_coords(canonical_signature, idx))
        variant_keys[idx] = key
        variant_mins[idx] = mins

    full_yellow: set[tuple[int, int]] = set()
    copy_frames: list[dict[str, object]] = []

    for item in per_copy:
        pink = np.asarray(item["pink"], dtype=int)
        yellow = np.asarray(item["yellow"], dtype=int)
        observed_key, _shifted, observed_mins = _normalize_coords(pink)
        matches = [idx for idx, key in variant_keys.items() if key == observed_key]
        assert len(matches) == 1, "signature must determine a unique orientation"
        idx = matches[0]
        offset = observed_mins - variant_mins[idx]

        local_yellow: set[tuple[int, int]] = set()
        for rr, cc in yellow:
            pr, pc = _inverse_transform_point((int(rr - offset[0]), int(cc - offset[1])), idx)
            local_yellow.add((pr, pc))
        full_yellow |= local_yellow
        copy_frames.append({"idx": idx, "offset": offset, "seen": local_yellow})

    out = g.copy()
    for frame in copy_frames:
        idx = int(frame["idx"])
        offset = np.asarray(frame["offset"], dtype=int)
        for point in full_yellow:
            rr, cc = _transform_point(point, idx)
            rr += int(offset[0])
            cc += int(offset[1])
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(out[rr, cc]) != PINK:
                out[rr, cc] = YELLOW
    return out


def _signature_is_asymmetric(signature: set[tuple[int, int]]) -> bool:
    keys = set(_all_transform_keys(signature).values())
    return len(keys) == 8


def _sample_signature() -> set[tuple[int, int]] | None:
    target = random.randint(5, 6)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = sorted({nb for cell in cells for nb in _neighbors4(cell) if nb not in cells})
        choices = []
        for cand in frontier:
            trial = set(cells)
            trial.add(cand)
            r0, c0, r1, c1 = _bbox_of(trial)
            if r1 - r0 + 1 <= 4 and c1 - c0 + 1 <= 4:
                choices.append(cand)
        if not choices:
            return None
        cells.add(random.choice(choices))
    if not _signature_is_asymmetric(cells):
        return None
    return cells


def _sample_template(copy_count: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    signature = _sample_signature()
    if signature is None:
        return None

    max_span = random.randint(7, 9)
    backbone_size = random.randint(5, 7)
    optional_size = random.randint(2 * copy_count + 1, 2 * copy_count + 4)

    body_core = set(signature)
    backbone: set[tuple[int, int]] = set()
    while len(backbone) < backbone_size:
        frontier = sorted({nb for cell in body_core for nb in _neighbors4(cell) if nb not in body_core})
        choices = []
        for cand in frontier:
            trial = body_core | {cand}
            r0, c0, r1, c1 = _bbox_of(trial)
            if r1 - r0 + 1 <= max_span and c1 - c0 + 1 <= max_span:
                choices.append(cand)
        if not choices:
            return None
        pick = random.choice(choices)
        backbone.add(pick)
        body_core.add(pick)

    base = set(signature) | backbone
    optional_candidates = sorted({nb for cell in base for nb in _neighbors4(cell) if nb not in base})
    random.shuffle(optional_candidates)
    optional: set[tuple[int, int]] = set()
    for cand in optional_candidates:
        trial = base | optional | {cand}
        r0, c0, r1, c1 = _bbox_of(trial)
        if r1 - r0 + 1 <= max_span and c1 - c0 + 1 <= max_span:
            optional.add(cand)
        if len(optional) >= optional_size:
            break
    if len(optional) < optional_size:
        return None

    yellow = backbone | optional
    return signature, backbone, yellow


def _sample_orientations(copy_count: int) -> list[int]:
    while True:
        choices = [random.randrange(8) for _ in range(copy_count)]
        if len(set(choices)) >= 2 and any(idx < 4 for idx in choices) and any(idx >= 4 for idx in choices):
            return choices


def _plan_visibility(yellow: set[tuple[int, int]], backbone: set[tuple[int, int]], copy_count: int) -> list[set[tuple[int, int]]] | None:
    optional = sorted(yellow - backbone)
    if len(optional) < 2 * copy_count:
        return None
    random.shuffle(optional)
    exclusives = optional[:copy_count]
    commons = optional[copy_count:]
    if len(commons) < copy_count:
        return None

    hidden = commons[:copy_count]
    visible = [set(backbone) | {exclusives[i]} for i in range(copy_count)]

    assignments = {cell: set() for cell in commons}
    for i, cell in enumerate(commons):
        allowed = [j for j in range(copy_count) if cell != hidden[j]]
        k = random.randint(1, max(1, min(copy_count - 1, len(allowed))))
        owners = set(random.sample(allowed, k))
        assignments[cell] = owners
        for owner in owners:
            visible[owner].add(cell)

    for i in range(copy_count):
        if hidden[i] in visible[i]:
            visible[i].remove(hidden[i])
        if visible[i] >= yellow:
            return None
        if yellow - visible[i] == set():
            return None

    union = set().union(*visible)
    if union != yellow:
        return None
    for i in range(copy_count):
        union_without = set().union(*(visible[j] for j in range(copy_count) if j != i))
        if union_without == yellow:
            return None
    return visible


def _normalize_transformed(cells: set[tuple[int, int]], idx: int) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]:
    arr = _transform_coords(np.asarray(sorted(cells), dtype=int), idx)
    mins = arr.min(axis=0)
    shifted = arr - mins
    placed = {(int(r), int(c)) for r, c in shifted}
    r0, c0, r1, c1 = _bbox_of(placed)
    return placed, (int(mins[0]), int(mins[1])), (r0, c0, r1, c1)


def _transform_with_reference(cells: set[tuple[int, int]], idx: int, ref_mins: tuple[int, int]) -> set[tuple[int, int]]:
    arr = _transform_coords(np.asarray(sorted(cells), dtype=int), idx)
    mins = np.asarray(ref_mins, dtype=int)
    shifted = arr - mins
    return {(int(r), int(c)) for r, c in shifted}


def _can_place(box: tuple[int, int, int, int], occupied: list[tuple[int, int, int, int]], padding: int) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in occupied:
        if not (r1 + padding < or0 or or1 + padding < r0 or c1 + padding < oc0 or oc1 + padding < c0):
            return False
    return True


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    copy_count = random.randint(3, 5)
    sampled = _sample_template(copy_count)
    if sampled is None:
        return None
    signature, backbone, yellow = sampled
    visibility = _plan_visibility(yellow, backbone, copy_count)
    if visibility is None:
        return None

    orientations = _sample_orientations(copy_count)
    full_template = signature | yellow

    variants = []
    for idx in orientations:
        full_norm, full_mins, full_box = _normalize_transformed(full_template, idx)
        sig_norm = _transform_with_reference(signature, idx, full_mins)
        y_all_norm = _transform_with_reference(yellow, idx, full_mins)
        variants.append(
            {
                "idx": idx,
                "full": full_norm,
                "full_mins": full_mins,
                "signature": sig_norm,
                "yellow_all": y_all_norm,
                "box": full_box,
            }
        )

    max_h = max(v["box"][2] + 1 for v in variants)
    max_w = max(v["box"][3] + 1 for v in variants)
    min_h = max(16, max_h + 4)
    min_w = max(16, max_w + 4)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(60):
        h = random.randint(min_h, min(30, max(min_h, min_h + 8)))
        w = random.randint(min_w, min(30, max(min_w, min_w + 8)))
        placed: list[dict[str, object]] = []
        occupied: list[tuple[int, int, int, int]] = []

        order = sorted(range(copy_count), key=lambda i: (variants[i]["box"][2] - variants[i]["box"][0] + 1) * (variants[i]["box"][3] - variants[i]["box"][1] + 1), reverse=True)
        ok = True
        for i in order:
            r0, c0, r1, c1 = variants[i]["box"]
            vh = r1 - r0 + 1
            vw = c1 - c0 + 1
            candidates = []
            for top in range(1, h - vh):
                for left in range(1, w - vw):
                    box = (top, left, top + vh - 1, left + vw - 1)
                    if _can_place(box, occupied, padding=1):
                        candidates.append((top, left))
            if not candidates:
                ok = False
                break
            top, left = random.choice(candidates)
            occupied.append((top, left, top + vh - 1, left + vw - 1))
            placed.append({"copy_index": i, "top": top, "left": left})
        if not ok:
            continue

        pos_by_copy = {item["copy_index"]: item for item in placed}
        inp = np.full((h, w), BLACK, dtype=int)
        out = np.full((h, w), BLACK, dtype=int)

        meta: list[dict[str, object]] = []
        for i in range(copy_count):
            top = int(pos_by_copy[i]["top"])
            left = int(pos_by_copy[i]["left"])
            variant = variants[i]
            visible_yellow = visibility[i]
            vis_norm = _transform_with_reference(visible_yellow, variant["idx"], variant["full_mins"])

            sig_abs = _shift_cells(variant["signature"], (top, left))
            vis_abs = _shift_cells(vis_norm, (top, left))
            full_y_abs = _shift_cells(variant["yellow_all"], (top, left))

            for r, c in sig_abs:
                inp[r, c] = PINK
                out[r, c] = PINK
            for r, c in vis_abs:
                inp[r, c] = YELLOW
            for r, c in full_y_abs:
                out[r, c] = YELLOW

            meta.append({"sig_abs": sig_abs, "visible": vis_abs, "full": full_y_abs})

        if np.array_equal(inp, out):
            continue
        if any(item["visible"] == item["full"] for item in meta):
            continue
        if len(_find_components(inp != BLACK)) != copy_count:
            continue

        for item in meta:
            if not item["visible"] < item["full"]:
                ok = False
                break
        if not ok:
            continue

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        added = {(r, c) for r in range(h) for c in range(w) if int(out[r, c]) == YELLOW and int(inp[r, c]) != YELLOW}
        if not added:
            continue
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_example()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate a valid complement-scroll input")

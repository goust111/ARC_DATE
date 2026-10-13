from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
TRANSFORMS = ("hflip", "vflip")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _label_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_transform(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.fliplr(arr)
    if kind == "vflip":
        return np.flipud(arr)
    raise ValueError(kind)


def _recolor(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def _inverse_mapping(mapping: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in mapping.items()}


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        for comp in _label_components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h != w or h < 4:
                continue
            if len(comp) != 2 * (h + w) - 4:
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(patch[0, :] == color)
                and np.all(patch[-1, :] == color)
                and np.all(patch[:, 0] == color)
                and np.all(patch[:, -1] == color)
            ):
                continue
            if np.any(patch[1:-1, 1:-1] == color):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "inner": patch[1:-1, 1:-1].copy(),
                }
            )
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))
    return frames


def _infer_rule_candidates(src: np.ndarray, dst: np.ndarray) -> list[tuple[str, dict[int, int]]]:
    candidates: list[tuple[str, dict[int, int]]] = []
    for kind in TRANSFORMS:
        moved = _apply_transform(src, kind)
        if np.any((moved == 0) != (dst == 0)):
            continue
        mapping: dict[int, int] = {}
        ok = True
        for a, b in zip(moved.ravel(), dst.ravel()):
            a = int(a)
            b = int(b)
            if a == 0:
                if b != 0:
                    ok = False
                    break
                continue
            if b == 0:
                ok = False
                break
            if a in mapping and mapping[a] != b:
                ok = False
                break
            mapping[a] = b
        if not ok or not mapping:
            continue
        vals = set(mapping.values())
        if vals != set(mapping.keys()):
            continue
        if any(mapping[mapping[k]] != k for k in mapping):
            continue
        candidates.append((kind, mapping))
    return candidates


def _infer_rule(src: np.ndarray, dst: np.ndarray) -> tuple[str, dict[int, int]]:
    candidates = _infer_rule_candidates(src, dst)
    if len(candidates) != 1:
        raise AssertionError(f"expected unique rule, got {candidates}")
    return candidates[0]


def _component_infos(inner: np.ndarray) -> list[dict]:
    infos = []
    for comp in _label_components(inner != 0):
        colors = sorted({int(inner[r, c]) for r, c in comp})
        r0, c0, r1, c1 = _bbox(comp)
        infos.append(
            {
                "coords": comp,
                "coord_set": set(comp),
                "bbox": (r0, c0, r1, c1),
                "colors": colors,
                "touch": any(r in (0, inner.shape[0] - 1) or c in (0, inner.shape[1] - 1) for r, c in comp),
            }
        )
    infos.sort(key=lambda info: (info["bbox"][0], info["bbox"][1]))
    return infos


def _transform_component(inner: np.ndarray, info: dict, kind: str, mapping: dict[int, int]) -> dict:
    r0, c0, r1, c1 = info["bbox"]
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in info["coords"]:
        patch[r - r0, c - c0] = inner[r, c]
    moved = _recolor(_apply_transform(patch, kind), mapping)
    coords = [(r0 + dr, c0 + dc) for dr, dc in np.argwhere(moved != 0)]
    colors = sorted(int(v) for v in np.unique(moved) if int(v) != 0)
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": moved,
        "coords": coords,
        "coord_set": set(coords),
        "color": colors[0] if len(colors) == 1 else None,
        "colors": colors,
    }


def _winner_for_non_touch(comp: dict, touch_infos: list[dict]) -> tuple[int, dict]:
    counts = {}
    for idx, touch in enumerate(touch_infos):
        count = 0
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in touch["coord_set"]:
                    count += 1
        counts[idx] = count
    best = max(counts.values())
    if best <= 0:
        raise AssertionError("non-touch component must contact a transformed touch component")
    winners = [idx for idx, count in counts.items() if count == best]
    winners.sort(key=lambda idx: touch_infos[idx]["bbox"][:2])
    return winners[0], {"counts": counts, "tied": len(winners) > 1}


def _winner_for_non_touch_optional(comp: dict, touch_infos: list[dict]) -> tuple[int, dict] | None:
    counts = {}
    for idx, touch in enumerate(touch_infos):
        count = 0
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in touch["coord_set"]:
                    count += 1
        counts[idx] = count
    best = max(counts.values(), default=0)
    if best <= 0:
        return None
    winners = [idx for idx, count in counts.items() if count == best]
    winners.sort(key=lambda idx: touch_infos[idx]["bbox"][:2])
    return winners[0], {"counts": counts, "tied": len(winners) > 1}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a flip-plus-color-swap rule from two training frames, apply it to touch-border components in the grey test frame, then propagate the resulting colors inward by side-contact majority.

    Input structure:
    - The grid has black background and exactly three hollow square frames with 1-cell borders.
    - Two non-grey frames form a training pair whose interiors show the same palette before and after a transformation.
    - The grey frame is the test frame; its interior contains several nonzero 4-connected components, some touching the inner border and some fully internal.

    Rule:
    - Detect the three square frames and take the two non-grey interiors as the training pair and the grey interior as the test case.
    - Infer the unique rule that turns one training interior into the other: either a horizontal flip or a vertical flip, followed by an involutive color mapping on the nonzero colors.
    - In the grey test interior, split all nonzero cells into 4-connected components.
    - Components touching the inner border are transformed directly: flip each one inside its own bbox, then recolor it by the inferred mapping.
    - Fully internal components are not flipped; instead, count their shared side contacts with each already-transformed touch-border component and recolor the whole component to the winning transformed component's color, breaking ties by the transformed touch component whose bbox starts earliest in row-major order.

    Key insight:
    - The training pair reveals a local rewrite rule, but only border-touching components obey it directly; interior components get their final color indirectly from whichever transformed border component they physically contact the most.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    if len(frames) != 3:
        raise AssertionError("expected exactly three frames")

    test_frames = [f for f in frames if f["color"] == GREY]
    if len(test_frames) != 1:
        raise AssertionError("expected exactly one grey test frame")
    test_frame = test_frames[0]
    train_frames = [f for f in frames if f["color"] != GREY]
    if len(train_frames) != 2:
        raise AssertionError("expected exactly two training frames")

    kind, mapping = _infer_rule(train_frames[0]["inner"], train_frames[1]["inner"])
    inner = test_frame["inner"]
    infos = _component_infos(inner)
    touch_infos = [info for info in infos if info["touch"]]
    non_touch_infos = [info for info in infos if not info["touch"]]
    if len(touch_infos) < 2 or len(non_touch_infos) < 2:
        raise AssertionError("expected at least two touch and two non-touch components")

    out = np.zeros_like(inner)
    transformed_touch = [_transform_component(inner, info, kind, mapping) for info in touch_infos]
    for touch in transformed_touch:
        r0, c0, r1, c1 = touch["bbox"]
        patch = touch["patch"]
        if np.any((out[r0 : r1 + 1, c0 : c1 + 1] != 0) & (patch != 0)):
            raise AssertionError("transformed touch components must stay disjoint")
        out[r0 : r1 + 1, c0 : c1 + 1][patch != 0] = patch[patch != 0]

    for info in non_touch_infos:
        winner_idx, _meta = _winner_for_non_touch(info, transformed_touch)
        winner_color = transformed_touch[winner_idx]["color"]
        if winner_color is None:
            raise AssertionError("touch components must become monochrome")
        for r, c in info["coords"]:
            out[r, c] = winner_color

    return out


BASE_TOUCH_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
]

VFLIP_LEFT_SHAPES = [
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0], [0, 1]], dtype=int),
]

INTERNAL_SHAPES = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
]


def _oriented_touch_shapes(mode: str, side: str) -> list[np.ndarray]:
    if mode == "hflip":
        if side == "top":
            return [shape.copy() for shape in BASE_TOUCH_SHAPES]
        return [np.flipud(shape) for shape in BASE_TOUCH_SHAPES]
    if side == "left":
        return [shape.copy() for shape in VFLIP_LEFT_SHAPES]
    return [np.fliplr(shape) for shape in VFLIP_LEFT_SHAPES]


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> set[tuple[int, int]]:
    coords = set()
    for dr, dc in np.argwhere(mask != 0):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        grid[rr, cc] = color
        coords.add((rr, cc))
    return coords


def _can_place_mask(occupied: set[tuple[int, int]], shape: tuple[int, int], r0: int, c0: int, outer_shape: tuple[int, int]) -> bool:
    h, w = shape
    if r0 < 0 or c0 < 0 or r0 + h > outer_shape[0] or c0 + w > outer_shape[1]:
        return False
    cells = {(r0 + int(dr), c0 + int(dc)) for dr in range(h) for dc in range(w)}
    if occupied & cells:
        return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _touch_patterns(mode: str) -> list[list[str]]:
    if mode == "hflip":
        return [["top", "top"], ["bottom", "bottom"]]
    return [["left", "left"], ["right", "right"]]


def _place_touch_output(mode: str, inner_shape: tuple[int, int], colors: list[int]) -> list[dict] | None:
    h, w = inner_shape
    sides = random.choice(_touch_patterns(mode))
    random.shuffle(colors)
    touch_colors = colors[: len(sides)]
    if len(set(touch_colors)) < 2:
        return None
    side = sides[0]
    shapes_a = _oriented_touch_shapes(mode, side)
    shapes_b = _oriented_touch_shapes(mode, side)
    random.shuffle(shapes_a)
    random.shuffle(shapes_b)
    infos: list[dict] = []
    for mask_a in shapes_a:
        for mask_b in shapes_b:
            h1, w1 = mask_a.shape
            h2, w2 = mask_b.shape
            gap = random.choice([1, 2])
            if side in ("top", "bottom"):
                total = w1 + gap + w2
                if total > w:
                    continue
                start = random.randint(0, w - total)
                positions = [(0 if side == "top" else h - h1, start), (0 if side == "top" else h - h2, start + w1 + gap)]
            else:
                total = h1 + gap + h2
                if total > h:
                    continue
                start = random.randint(0, h - total)
                positions = [(start, 0 if side == "left" else w - w1), (start + h1 + gap, 0 if side == "left" else w - w2)]
            infos = []
            for (r0, c0), mask, color in zip(positions, (mask_a, mask_b), touch_colors):
                mh, mw = mask.shape
                coords = {(r0 + int(dr), c0 + int(dc)) for dr, dc in np.argwhere(mask != 0)}
                infos.append(
                    {
                        "bbox": (r0, c0, r0 + mh - 1, c0 + mw - 1),
                        "coords": sorted(coords),
                        "coord_set": coords,
                        "color": color,
                        "side": side,
                        "mask": mask.copy(),
                    }
                )
            return infos
    return None


def _counts_for_cells(cells: set[tuple[int, int]], touch_infos: list[dict]) -> dict[int, int]:
    counts = {}
    for idx, touch in enumerate(touch_infos):
        cnt = 0
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in touch["coord_set"]:
                    cnt += 1
        counts[idx] = cnt
    return counts


def _winner_from_counts(counts: dict[int, int], touch_infos: list[dict]) -> tuple[int, bool]:
    best = max(counts.values())
    if best <= 0:
        raise AssertionError("expected a positive contact count")
    winners = [idx for idx, count in counts.items() if count == best]
    winners.sort(key=lambda idx: touch_infos[idx]["bbox"][:2])
    return winners[0], len(winners) > 1


def _enumerate_internal_candidates(inner_shape: tuple[int, int], touch_infos: list[dict]) -> list[dict]:
    h, w = inner_shape
    occupied = set().union(*(touch["coord_set"] for touch in touch_infos))
    candidates: list[dict] = []
    seen_cells: set[frozenset[tuple[int, int]]] = set()
    for base in INTERNAL_SHAPES:
        variants = []
        for k in range(4):
            rot = np.rot90(base, k)
            key = rot.tobytes()
            if all(key != v.tobytes() for v in variants):
                variants.append(rot)
        for mask in variants:
            mh, mw = mask.shape
            for r0 in range(1, h - mh):
                for c0 in range(1, w - mw):
                    cells = {(r0 + int(dr), c0 + int(dc)) for dr, dc in np.argwhere(mask != 0)}
                    if cells & occupied:
                        continue
                    key = frozenset(cells)
                    if key in seen_cells:
                        continue
                    counts = _counts_for_cells(cells, touch_infos)
                    contributors = sum(v > 0 for v in counts.values())
                    if contributors < 1:
                        continue
                    winner, tied = _winner_from_counts(counts, touch_infos)
                    seen_cells.add(key)
                    candidates.append(
                        {
                            "cells": cells,
                            "bbox": (r0, c0, r0 + mh - 1, c0 + mw - 1),
                            "counts": counts,
                            "contributors": contributors,
                            "winner": winner,
                            "tied": tied,
                        }
                    )
    candidates.sort(key=lambda cand: (cand["bbox"][0], cand["bbox"][1]))
    return candidates


def _non_touch_compatible(chosen: list[dict], cand: dict) -> bool:
    cells = cand["cells"]
    for other in chosen:
        if cells & other["cells"]:
            return False
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in other["cells"]:
                    return False
    return True


def _choose_internal_set(candidates: list[dict], touch_infos: list[dict]) -> list[dict] | None:
    need_tie = random.random() < 0.3
    target_count = 2
    random.shuffle(candidates)
    chosen: list[dict] = []

    def rec(idx: int) -> list[dict] | None:
        if len(chosen) == target_count:
            if need_tie and not any(cand["tied"] for cand in chosen):
                return None
            profiles = {tuple(cand["counts"][k] for k in sorted(cand["counts"])) for cand in chosen}
            if len(profiles) < 2 and len(chosen) > 1:
                return None
            if not any(cand["contributors"] >= 2 for cand in chosen):
                return None
            return [cand.copy() for cand in chosen]
        if idx >= len(candidates):
            return None
        for j in range(idx, len(candidates)):
            cand = candidates[j]
            if not _non_touch_compatible(chosen, cand):
                continue
            chosen.append(cand)
            found = rec(j + 1)
            if found is not None:
                return found
            chosen.pop()
        return None

    return rec(0)


def _swap_mapping(palette: list[int]) -> dict[int, int]:
    shuffled = palette[:]
    random.shuffle(shuffled)
    mapping = {}
    for i in range(0, len(shuffled), 2):
        a = shuffled[i]
        b = shuffled[i + 1]
        mapping[a] = b
        mapping[b] = a
    return mapping


def _build_training_pair(kind: str, mapping: dict[int, int], palette: list[int], size: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(120):
        src = np.zeros((size, size), dtype=int)
        occupied: set[tuple[int, int]] = set()
        colors = palette[:]
        random.shuffle(colors)
        n_components = random.randint(max(3, len(palette)), max(4, len(palette)))
        all_shapes = INTERNAL_SHAPES + [np.array([[1, 1, 0], [0, 1, 1]], dtype=int)]
        placed = 0
        while placed < n_components:
            mask = random.choice(all_shapes)
            if random.random() < 0.5:
                mask = np.rot90(mask, random.randint(0, 3))
            mh, mw = mask.shape
            positions = [(r0, c0) for r0 in range(0, size - mh + 1) for c0 in range(0, size - mw + 1)]
            random.shuffle(positions)
            color = colors[placed % len(colors)]
            did_place = False
            for r0, c0 in positions:
                cells = {(r0 + int(dr), c0 + int(dc)) for dr, dc in np.argwhere(mask != 0)}
                if cells & occupied:
                    continue
                if any((nr, nc) in occupied for r, c in cells for nr, nc in _neighbors4(r, c)):
                    continue
                for r, c in cells:
                    src[r, c] = color
                occupied |= cells
                placed += 1
                did_place = True
                break
            if not did_place:
                break
        if placed < n_components:
            continue
        if set(int(v) for v in np.unique(src) if int(v) != 0) != set(palette):
            continue
        dst = _recolor(_apply_transform(src, kind), mapping)
        if np.array_equal(src, dst):
            continue
        if len(_infer_rule_candidates(src, dst)) != 1:
            continue
        return src, dst
    return None


def _build_test_case(kind: str, mapping: dict[int, int], palette: list[int], inner_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    mapped_palette = list({mapping[c] for c in palette})
    for _ in range(160):
        touch_infos = _place_touch_output(kind, inner_shape, mapped_palette[:])
        if touch_infos is None or len(touch_infos) < 2:
            continue
        candidates = _enumerate_internal_candidates(inner_shape, touch_infos)
        chosen_internal = _choose_internal_set(candidates, touch_infos)
        if chosen_internal is None:
            continue

        out_inner = np.zeros(inner_shape, dtype=int)
        for touch in touch_infos:
            _paint_mask(out_inner, touch["mask"], touch["bbox"][0], touch["bbox"][1], touch["color"])

        for comp in chosen_internal:
            color = touch_infos[comp["winner"]]["color"]
            for r, c in comp["cells"]:
                out_inner[r, c] = color

        inv_mapping = _inverse_mapping(mapping)
        inp_inner = np.zeros(inner_shape, dtype=int)
        for touch in touch_infos:
            r0, c0, r1, c1 = touch["bbox"]
            patch_out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            for dr, dc in np.argwhere(touch["mask"] != 0):
                patch_out[int(dr), int(dc)] = touch["color"]
            patch_in = _apply_transform(_recolor(patch_out, inv_mapping), kind)
            inp_inner[r0 : r1 + 1, c0 : c1 + 1][patch_in != 0] = patch_in[patch_in != 0]

        changed_internal = False
        for comp in chosen_internal:
            winner_color = touch_infos[comp["winner"]]["color"]
            options = [c for c in palette if mapping[c] != winner_color]
            if not options:
                options = palette[:]
            src_color = random.choice(options)
            changed_internal |= mapping[src_color] != winner_color
            for r, c in comp["cells"]:
                inp_inner[r, c] = src_color
        if not changed_internal:
            continue

        infos = _component_infos(inp_inner)
        touch_inputs = [info for info in infos if info["touch"]]
        non_touch_inputs = [info for info in infos if not info["touch"]]
        if len(touch_inputs) != len(touch_infos) or len(non_touch_inputs) != len(chosen_internal):
            continue
        if any(len(info["colors"]) != 1 for info in infos):
            continue

        true_winners = []
        valid_contacts = True
        pseudo_touch_infos = []
        for info in touch_inputs:
            tinfo = _transform_component(inp_inner, info, kind, mapping)
            if tinfo["color"] is None:
                valid_contacts = False
                break
            pseudo_touch_infos.append(tinfo)
        if not valid_contacts:
            continue
        if len(pseudo_touch_infos) != len(touch_infos):
            continue
        for info in non_touch_inputs:
            true_result = _winner_for_non_touch_optional(info, pseudo_touch_infos)
            if true_result is None:
                valid_contacts = False
                break
            winner_idx, meta = true_result
            true_winners.append((winner_idx, meta["counts"]))
        if not valid_contacts:
            continue

        no_flip_touch_infos = []
        for info in touch_inputs:
            r0, c0, r1, c1 = info["bbox"]
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            for r, c in info["coords"]:
                patch[r - r0, c - c0] = inp_inner[r, c]
            recolored = _recolor(patch, mapping)
            coords = {(r0 + int(dr), c0 + int(dc)) for dr, dc in np.argwhere(recolored != 0)}
            colors = sorted(int(v) for v in np.unique(recolored) if int(v) != 0)
            no_flip_touch_infos.append(
                {
                    "bbox": info["bbox"],
                    "coord_set": coords,
                    "color": colors[0] if len(colors) == 1 else None,
                }
            )

        flip_matters = False
        idx = 0
        for info in non_touch_inputs:
            no_flip_result = _winner_for_non_touch_optional(info, no_flip_touch_infos)
            if no_flip_result is None:
                flip_matters = True
                break
            winner_no_flip, _meta = no_flip_result
            if winner_no_flip != true_winners[idx][0]:
                flip_matters = True
                break
            if any(true_winners[idx][1][k] != _meta["counts"][k] for k in true_winners[idx][1]):
                flip_matters = True
                break
            idx += 1
        if not flip_matters:
            continue

        return inp_inner, out_inner
    return None


def _frame_canvas(h: int, w: int, color: int, inner: np.ndarray) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    out[0, :] = color
    out[-1, :] = color
    out[:, 0] = color
    out[:, -1] = color
    out[1:-1, 1:-1] = inner
    return out


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    if np.any(grid[r0 : r0 + h, c0 : c0 + w] != 0):
        raise AssertionError("patch overlap")
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _wrap_test_only(inner: np.ndarray) -> np.ndarray:
    patch = _frame_canvas(inner.shape[0] + 2, inner.shape[1] + 2, GREY, inner)
    return patch


def generate_input() -> np.ndarray:
    for _ in range(400):
        kind = random.choice(list(TRANSFORMS))
        palette_size = random.choice([2, 4])
        available = [c for c in range(1, 10) if c != GREY]
        palette = random.sample(available, palette_size)
        mapping = _swap_mapping(palette)

        train_size = random.randint(5, 6)
        training = _build_training_pair(kind, mapping, palette, train_size)
        if training is None:
            continue
        train_src, train_dst = training

        side = random.randint(8, 10)
        inner_shape = (side, side)
        test_case = _build_test_case(kind, mapping, palette, inner_shape)
        if test_case is None:
            continue
        test_inner, expected = test_case

        border_choices = [c for c in range(1, 10) if c not in set(palette) | {GREY}]
        if len(border_choices) < 2:
            continue
        border_a, border_b = random.sample(border_choices, 2)

        train_a = _frame_canvas(train_size + 2, train_size + 2, border_a, train_src)
        train_b = _frame_canvas(train_size + 2, train_size + 2, border_b, train_dst)
        test_patch = _frame_canvas(inner_shape[0] + 2, inner_shape[1] + 2, GREY, test_inner)

        top_gap = random.randint(2, 3)
        side_gap = random.randint(2, 3)
        middle_gap = random.randint(2, 4)
        bottom_gap = random.randint(2, 3)
        canvas_h = train_a.shape[0] + top_gap + test_patch.shape[0] + bottom_gap
        canvas_w = max(train_a.shape[1] + middle_gap + train_b.shape[1] + 2 * side_gap, test_patch.shape[1] + 2 * side_gap)
        if canvas_h > 30 or canvas_w > 30:
            continue

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        r_train = 1
        c_train_a = side_gap - 1
        c_train_b = canvas_w - side_gap - train_b.shape[1] + 1
        r_test = train_a.shape[0] + top_gap
        c_test = (canvas_w - test_patch.shape[1]) // 2
        try:
            _paste_patch(grid, train_a, r_train, c_train_a)
            _paste_patch(grid, train_b, r_train, c_train_b)
            _paste_patch(grid, test_patch, r_test, c_test)
        except AssertionError:
            continue

        if max(grid.shape) > 30:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        if len(_find_frames(grid)) != 3:
            continue
        return grid
    raise RuntimeError("failed to generate a valid example")

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, get_bg, label_mask, paste, random_free_location


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver helpers ---------------


def _edge_component_ids(labels: np.ndarray, n_labels: int) -> set[int]:
    if n_labels == 0:
        return set()
    edge_ids = set(map(int, labels[0, :]))
    edge_ids.update(map(int, labels[-1, :]))
    edge_ids.update(map(int, labels[:, 0]))
    edge_ids.update(map(int, labels[:, -1]))
    edge_ids.discard(0)
    return edge_ids


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0, c0, r1, c1 = bbox_coords(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    cropped = _crop_mask(mask).astype(np.uint8)
    return (int(cropped.shape[0]), int(cropped.shape[1]), cropped.tobytes())


def _extract_ring_infos(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    fg_labels, n_fg = label_mask(g != bg, connectivity=4)
    bg_labels, n_bg = label_mask(g == bg, connectivity=4)
    assert n_fg >= 1, "expected at least one foreground ring"

    edge_bg_ids = _edge_component_ids(bg_labels, n_bg)
    infos: dict[int, dict] = {}
    for obj_id in range(1, n_fg + 1):
        obj_mask = fg_labels == obj_id
        coords = np.argwhere(obj_mask)
        color = int(g[coords[0, 0], coords[0, 1]])
        infos[obj_id] = {
            "id": int(obj_id),
            "color": color,
            "coords": coords,
            "obj_mask": obj_mask,
            "hole_mask": None,
        }

    for bg_id in range(1, n_bg + 1):
        if bg_id in edge_bg_ids:
            continue

        hole_coords = np.argwhere(bg_labels == bg_id)
        owners: set[int] = set()
        for r, c in hole_coords:
            rr = int(r)
            cc = int(c)
            for dr, dc in _DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and fg_labels[nr, nc] != 0:
                    owners.add(int(fg_labels[nr, nc]))

        assert len(owners) == 1, "each enclosed hole must belong to exactly one ring"
        obj_id = next(iter(owners))
        assert infos[obj_id]["hole_mask"] is None, "each ring must enclose exactly one hole"
        hole_mask = np.zeros(g.shape, dtype=bool)
        hole_mask[hole_coords[:, 0], hole_coords[:, 1]] = True
        infos[obj_id]["hole_mask"] = hole_mask

    result = list(infos.values())
    assert all(info["hole_mask"] is not None for info in result), "every foreground object must be a ring"
    for info in result:
        hole = _crop_mask(info["hole_mask"])
        info["hole_shape"] = hole
        info["hole_key"] = _mask_key(hole)
        info["mirror_key"] = _mask_key(np.fliplr(hole))
    return result


def _pair_ring_ids(infos: list[dict]) -> list[tuple[int, int]]:
    by_id = {int(info["id"]): info for info in infos}
    candidate_partner: dict[int, int] = {}
    for info in infos:
        matches = [
            int(other["id"])
            for other in infos
            if int(other["id"]) != int(info["id"]) and other["hole_key"] == info["mirror_key"]
        ]
        assert len(matches) <= 1, "mirror-hole pairing must be unambiguous"
        if matches:
            candidate_partner[int(info["id"])] = matches[0]

    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    for obj_id, partner_id in candidate_partner.items():
        if obj_id in seen:
            continue
        assert candidate_partner.get(partner_id) == obj_id, "mirror pairing must be mutual"
        seen.add(obj_id)
        seen.add(partner_id)
        a = min(obj_id, partner_id)
        b = max(obj_id, partner_id)
        pairs.append((a, b))

    assert len(seen) == 2 * len(pairs)
    assert set(by_id).issuperset(seen)
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair ring objects by mirrored hole shapes and swap the paired shell colors.

    Input structure:
    - A black background contains several disconnected monochromatic ring objects of distinct colors.
    - Each ring encloses exactly one black hole, and every hole stays away from the grid border.
    - Some hole shapes have exactly one left-right mirrored partner elsewhere, while other rings are unmatched distractors.

    Rule:
    - Extract each ring's enclosed black hole shape, normalized up to translation.
    - Pair two rings iff one hole shape equals the other hole shape after a horizontal mirror.
    - Swap the shell colors inside every matched pair and leave unmatched rings unchanged.

    Key insight:
    - The matching cue is the negative space inside each ring; outer shell geometry, scale, and position are deliberate distractions.
    """
    g = np.asarray(grid, dtype=int)
    infos = _extract_ring_infos(g)
    pairs = _pair_ring_ids(infos)

    out = g.copy()
    by_id = {int(info["id"]): info for info in infos}
    for left_id, right_id in pairs:
        left = by_id[left_id]
        right = by_id[right_id]
        out[left["obj_mask"]] = int(right["color"])
        out[right["obj_mask"]] = int(left["color"])
    return out


# --------------- generator ---------------

_GEN_MAX_ATTEMPTS = 400
_GEN_HOLE_ATTEMPTS = 80
_GEN_PAIR_COUNT_CHOICES = (1, 1, 2, 2, 3)
_GEN_SINGLE_COUNT_CHOICES = (0, 1, 1, 2, 2, 3)
_GEN_HOLE_H_RANGE = (2, 5)
_GEN_HOLE_W_RANGE = (2, 6)
_GEN_HOLE_AREA_RATIO_RANGE = (0.38, 0.72)
_GEN_INNER_MARGIN_RANGE = (1, 3)
_GEN_PROTRUSION_COUNT_RANGE = (1, 4)
_GEN_PROTRUSION_THICKNESS_RANGE = (1, 3)
_GEN_CANVAS_MIN = 14
_GEN_CANVAS_MAX = 30
_GEN_CLEARANCE = 1
_GEN_RING_PATCH_ATTEMPTS = 80


def _gen_connected_mask(
    rng: np.random.Generator,
    *,
    require_asymmetric: bool,
) -> np.ndarray | None:
    for _ in range(_GEN_HOLE_ATTEMPTS):
        h = int(rng.integers(_GEN_HOLE_H_RANGE[0], _GEN_HOLE_H_RANGE[1] + 1))
        w = int(rng.integers(_GEN_HOLE_W_RANGE[0], _GEN_HOLE_W_RANGE[1] + 1))
        if require_asymmetric and w == 1:
            continue

        min_area = max(3, int(np.ceil(h * w * _GEN_HOLE_AREA_RATIO_RANGE[0])))
        max_area = max(min_area, int(np.floor(h * w * _GEN_HOLE_AREA_RATIO_RANGE[1])))
        if min_area > max_area:
            continue
        target_area = int(rng.integers(min_area, max_area + 1))

        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {seed}
        while len(cells) < target_area:
            frontier = sorted(
                {
                    (r + dr, c + dc)
                    for r, c in cells
                    for dr, dc in _DIRS4
                    if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
                }
            )
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != target_area:
            continue

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if min(mask.shape) < 2:
            continue
        if require_asymmetric and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _gen_unique_hole(
    rng: np.random.Generator,
    used_keys: set[tuple[int, int, bytes]],
    *,
    require_asymmetric: bool,
) -> np.ndarray | None:
    for _ in range(_GEN_HOLE_ATTEMPTS):
        mask = _gen_connected_mask(rng, require_asymmetric=require_asymmetric)
        if mask is None:
            continue
        key = _mask_key(mask)
        mirror_key = _mask_key(np.fliplr(mask))
        if key in used_keys or mirror_key in used_keys:
            continue
        used_keys.add(key)
        used_keys.add(mirror_key)
        return mask
    return None


def _is_valid_ring_patch(patch: np.ndarray, color: int) -> bool:
    fg_labels, n_fg = label_mask(patch == int(color), connectivity=4)
    if n_fg != 1:
        return False
    bg_labels, n_bg = label_mask(patch == Color.BLACK, connectivity=4)
    edge_bg_ids = _edge_component_ids(bg_labels, n_bg)
    inner_holes = [bg_id for bg_id in range(1, n_bg + 1) if bg_id not in edge_bg_ids]
    return len(inner_holes) == 1


def _gen_ring_patch(rng: np.random.Generator, hole_mask: np.ndarray, color: int) -> np.ndarray | None:
    hole_h, hole_w = map(int, hole_mask.shape)
    for _ in range(_GEN_RING_PATCH_ATTEMPTS):
        pad_top = int(rng.integers(_GEN_INNER_MARGIN_RANGE[0], _GEN_INNER_MARGIN_RANGE[1] + 1))
        pad_bottom = int(rng.integers(_GEN_INNER_MARGIN_RANGE[0], _GEN_INNER_MARGIN_RANGE[1] + 1))
        pad_left = int(rng.integers(_GEN_INNER_MARGIN_RANGE[0], _GEN_INNER_MARGIN_RANGE[1] + 1))
        pad_right = int(rng.integers(_GEN_INNER_MARGIN_RANGE[0], _GEN_INNER_MARGIN_RANGE[1] + 1))

        base_h = hole_h + pad_top + pad_bottom
        base_w = hole_w + pad_left + pad_right
        canvas_h = base_h + 8
        canvas_w = base_w + 8
        outer = np.zeros((canvas_h, canvas_w), dtype=bool)

        base_r0 = 4
        base_c0 = 4
        outer[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

        protrusion_count = int(
            rng.integers(_GEN_PROTRUSION_COUNT_RANGE[0], _GEN_PROTRUSION_COUNT_RANGE[1] + 1)
        )
        for _ in range(protrusion_count):
            side = int(rng.integers(0, 4))
            thick = int(
                rng.integers(
                    _GEN_PROTRUSION_THICKNESS_RANGE[0],
                    _GEN_PROTRUSION_THICKNESS_RANGE[1] + 1,
                )
            )
            if side in (0, 1):
                span = int(rng.integers(1, base_w + 1))
                c0 = int(rng.integers(base_c0, base_c0 + base_w - span + 1))
                if side == 0:
                    outer[base_r0 - thick : base_r0, c0 : c0 + span] = True
                else:
                    outer[base_r0 + base_h : base_r0 + base_h + thick, c0 : c0 + span] = True
            else:
                span = int(rng.integers(1, base_h + 1))
                r0 = int(rng.integers(base_r0, base_r0 + base_h - span + 1))
                if side == 2:
                    outer[r0 : r0 + span, base_c0 - thick : base_c0] = True
                else:
                    outer[r0 : r0 + span, base_c0 + base_w : base_c0 + base_w + thick] = True

        patch = np.zeros_like(outer, dtype=int)
        patch[outer] = int(color)
        hole_r0 = base_r0 + pad_top
        hole_c0 = base_c0 + pad_left
        hole_view = patch[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w]
        hole_view[hole_mask] = Color.BLACK

        colored = np.argwhere(patch != Color.BLACK)
        r0, c0, r1, c1 = bbox_coords(colored)
        patch = np.asarray(patch[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        if _is_valid_ring_patch(patch, color):
            return patch
    return None


def _gen_counts(rng: np.random.Generator) -> tuple[int, int] | None:
    pair_count = int(_GEN_PAIR_COUNT_CHOICES[int(rng.integers(0, len(_GEN_PAIR_COUNT_CHOICES)))])
    single_options = [count for count in _GEN_SINGLE_COUNT_CHOICES if 2 * pair_count + count <= len(Color.NOT_BLACK)]
    if not single_options:
        return None
    single_count = int(single_options[int(rng.integers(0, len(single_options)))])
    return pair_count, single_count


def _gen_canvas_shape(rng: np.random.Generator, patches: list[np.ndarray]) -> tuple[int, int] | None:
    min_h = max(_GEN_CANVAS_MIN, max(int(p.shape[0]) for p in patches) + 2)
    min_w = max(_GEN_CANVAS_MIN, max(int(p.shape[1]) for p in patches) + 2)
    area_need = sum((int(p.shape[0]) + 2) * (int(p.shape[1]) + 2) for p in patches)
    tight = [
        (h, w)
        for h in range(min_h, _GEN_CANVAS_MAX + 1)
        for w in range(min_w, _GEN_CANVAS_MAX + 1)
        if area_need <= h * w <= max(area_need * 3, _GEN_CANVAS_MIN * _GEN_CANVAS_MIN)
    ]
    all_choices = [
        (h, w)
        for h in range(min_h, _GEN_CANVAS_MAX + 1)
        for w in range(min_w, _GEN_CANVAS_MAX + 1)
        if h * w >= area_need
    ]
    choices = tight if tight else all_choices
    if not choices:
        return None
    return choices[int(rng.integers(0, len(choices)))]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    counts = _gen_counts(rng)
    if counts is None:
        return None
    pair_count, single_count = counts

    used_keys: set[tuple[int, int, bytes]] = set()
    hole_groups: list[tuple[np.ndarray, int]] = []
    for pair_id in range(pair_count):
        base = _gen_unique_hole(rng, used_keys, require_asymmetric=True)
        if base is None:
            return None
        mirror = np.fliplr(base)
        hole_groups.append((base, pair_id))
        hole_groups.append((mirror, pair_id))

    for single_id in range(single_count):
        hole = _gen_unique_hole(rng, used_keys, require_asymmetric=bool(rng.integers(0, 2)))
        if hole is None:
            return None
        hole_groups.append((hole, pair_count + single_id))

    rng.shuffle(hole_groups)
    colors = rng.choice(np.array(Color.NOT_BLACK, dtype=int), size=len(hole_groups), replace=False)

    patches: list[tuple[np.ndarray, int]] = []
    for idx, (hole_mask, _group) in enumerate(hole_groups):
        patch = _gen_ring_patch(rng, hole_mask, int(colors[idx]))
        if patch is None:
            return None
        patches.append((patch, idx))

    shape = _gen_canvas_shape(rng, [patch for patch, _idx in patches])
    if shape is None:
        return None
    height, width = shape
    grid = np.full((height, width), Color.BLACK, dtype=int)
    blocked = np.zeros((height, width), dtype=int)

    patches.sort(key=lambda item: int(item[0].size), reverse=True)
    for patch, _idx in patches:
        block_patch = np.ones_like(patch, dtype=int)
        try:
            r0, c0 = random_free_location(
                blocked,
                block_patch,
                transparent=0,
                border=1,
                padding=_GEN_CLEARANCE,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=Color.BLACK)
        paste(blocked, block_patch, r0, c0, transparent=0)

    infos = _extract_ring_infos(grid)
    pairs = _pair_ring_ids(infos)
    if len(infos) != len(hole_groups):
        return None
    if len(pairs) != pair_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid negative-space pairing input")

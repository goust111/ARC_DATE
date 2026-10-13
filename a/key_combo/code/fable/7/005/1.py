from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, components_mask, crop_bbox, get_bg, label_mask, paste


BG = Color.BLACK
RING = Color.GREY
TEMPLATE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

GRID_MIN = 18
GRID_MAX = 30
PAIR_COUNT_CHOICES = [1, 2, 3, 4]
PAIR_COUNT_WEIGHTS = [0.18, 0.34, 0.30, 0.18]
SHAPE_MIN_AREA = 4
SHAPE_MAX_AREA = 10
SHAPE_MIN_SIDE = 2
SHAPE_MAX_SIDE = 6
PLACEMENT_PAD = 2
GEN_RETRY_LIMIT = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _edge_ids(labels: np.ndarray, n_labels: int) -> set[int]:
    if n_labels == 0:
        return set()
    ids = set(map(int, labels[0, :]))
    ids.update(map(int, labels[-1, :]))
    ids.update(map(int, labels[:, 0]))
    ids.update(map(int, labels[:, -1]))
    ids.discard(0)
    return ids


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _canonical_shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _mask_has_internal_hole(mask: np.ndarray) -> bool:
    padded = np.pad(~mask, 1, constant_values=True)
    labels, n_labels = label_mask(padded, connectivity=4)
    return n_labels > 1


def _random_connected_shape() -> np.ndarray:
    for _ in range(200):
        height = random.randint(SHAPE_MIN_SIDE, SHAPE_MAX_SIDE)
        width = random.randint(SHAPE_MIN_SIDE, SHAPE_MAX_SIDE)
        area_cap = min(SHAPE_MAX_AREA, height * width - 1)
        if area_cap < SHAPE_MIN_AREA:
            continue
        target_area = random.randint(SHAPE_MIN_AREA, area_cap)

        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue

        coords = np.array(sorted(cells), dtype=int)
        mask = _coords_to_mask(coords)
        if min(mask.shape) < SHAPE_MIN_SIDE:
            continue
        if not _is_rotation_asymmetric(mask):
            continue
        if _mask_has_internal_hole(mask):
            continue
        return mask
    return None


def _make_template_patch(shape_mask: np.ndarray) -> np.ndarray:
    area = int(shape_mask.sum())
    n_colors = random.randint(2, min(4, area))
    palette = random.sample(TEMPLATE_COLORS, n_colors)
    filled = [tuple(map(int, coord)) for coord in np.argwhere(shape_mask)]
    random.shuffle(filled)

    patch = np.zeros(shape_mask.shape, dtype=int)
    for idx, (r, c) in enumerate(filled):
        patch[r, c] = palette[(2 * int(r) + 3 * int(c) + idx) % n_colors]

    used = {int(v) for v in patch[shape_mask]}
    if len(used) < 2:
        patch[filled[0][0], filled[0][1]] = palette[0]
        patch[filled[-1][0], filled[-1][1]] = palette[1]
    return patch


def _make_ring_patch(hole_mask: np.ndarray) -> np.ndarray:
    h, w = map(int, hole_mask.shape)
    padded_hole = np.zeros((h + 2, w + 2), dtype=bool)
    padded_hole[1 : h + 1, 1 : w + 1] = hole_mask

    shell = np.zeros_like(padded_hole)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src_r0 = max(0, -dr)
            src_r1 = min(h + 2, h + 2 - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w + 2, w + 2 - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h + 2, h + 2 + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w + 2, w + 2 + dc)
            shell[dst_r0:dst_r1, dst_c0:dst_c1] |= padded_hole[src_r0:src_r1, src_c0:src_c1]
    shell &= ~padded_hole

    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[shell] = RING
    return patch


def _collect_holes(grid: np.ndarray) -> list[dict]:
    labels, n_labels = label_mask(grid == BG, connectivity=4)
    gray_labels, _ = label_mask(grid == RING, connectivity=8)
    edge_black = _edge_ids(labels, n_labels)
    holes = []
    height, width = grid.shape

    for comp_id in range(1, n_labels + 1):
        if comp_id in edge_black:
            continue
        coords = np.argwhere(labels == comp_id)
        gray_ids = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and gray_labels[nr, nc]:
                        gray_ids.add(int(gray_labels[nr, nc]))
        assert len(gray_ids) == 1, "each hole must be enclosed by exactly one gray ring"
        hole_mask = _coords_to_mask(coords)
        r0, c0, r1, c1 = bbox_coords(coords)
        holes.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": hole_mask,
                "ring_id": next(iter(gray_ids)),
            }
        )

    holes.sort(key=lambda item: item["bbox"])
    return holes


def _collect_templates(grid: np.ndarray) -> list[dict]:
    template_mask = (grid != BG) & (grid != RING)
    comps = components_mask(template_mask, connectivity=4)
    templates = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))
        templates.append(
            {
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "mask": patch != BG,
            }
        )
    templates.sort(key=lambda item: item["bbox"])
    return templates


def _solve_assignment(options_by_hole: list[list[tuple[int, int]]], n_templates: int) -> dict[int, tuple[int, int]]:
    order = sorted(range(len(options_by_hole)), key=lambda idx: len(options_by_hole[idx]))
    solutions: list[dict[int, tuple[int, int]]] = []

    def dfs(pos: int, used: set[int], partial: dict[int, tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            if len(used) == n_templates:
                solutions.append(dict(partial))
            return

        hole_idx = order[pos]
        for template_idx, rot_k in options_by_hole[hole_idx]:
            if template_idx in used:
                continue
            used.add(template_idx)
            partial[hole_idx] = (template_idx, rot_k)
            dfs(pos + 1, used, partial)
            used.remove(template_idx)
            partial.pop(hole_idx, None)

    dfs(0, set(), {})
    assert len(solutions) == 1, "template-hole matching must be unique"
    return solutions[0]


def _match_holes_to_templates(holes: list[dict], templates: list[dict]) -> dict[int, tuple[int, int]]:
    assert len(holes) == len(templates) >= 1
    options_by_hole: list[list[tuple[int, int]]] = []
    for hole in holes:
        options = []
        for template_idx, template in enumerate(templates):
            for rot_k in range(4):
                if np.array_equal(hole["mask"], np.rot90(template["mask"], rot_k)):
                    options.append((template_idx, rot_k))
        assert options, "every hole must match at least one template"
        options_by_hole.append(options)
    return _solve_assignment(options_by_hole, len(templates))


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Transplant each gray-ring hole by rotating its uniquely matching colored template into the hole and erase the source template.

    Input structure:
    - A black background contains one-pixel-thick gray closed rings, each surrounding one black hole component.
    - Elsewhere on the canvas are several disconnected multicolor solid template objects, none using gray or black.
    - Each hole shape matches exactly one template outline under one of the four right-angle rotations.

    Rule:
    - Detect every black hole as a non-edge black connected component enclosed by gray.
    - Compare the normalized hole mask against each template outline under rotations 0/90/180/270 and find the unique bijective matching.
    - Remove all source templates, keep the gray rings unchanged, and paste each matched template rotated into its corresponding hole.

    Key insight:
    - The shape clue is negative space: the black hole itself must be matched to a template's occupied silhouette, then the full multicolor interior pattern is rotated and transplanted.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    assert max(grid.shape) <= 30, "ARC grids must be at most 30x30"
    assert get_bg(grid) == BG, "expected black background"

    holes = _collect_holes(grid)
    templates = _collect_templates(grid)
    assignment = _match_holes_to_templates(holes, templates)

    out = grid.copy()
    out[(out != BG) & (out != RING)] = BG

    for hole_idx, hole in enumerate(holes):
        template_idx, rot_k = assignment[hole_idx]
        template_patch = templates[template_idx]["patch"]
        rotated = np.rot90(template_patch, rot_k)
        r0, c0, r1, c1 = hole["bbox"]
        assert rotated.shape == (r1 - r0 + 1, c1 - c0 + 1)
        paste(out, rotated, r0, c0, transparent=BG, overwrite=True)

    return out


def _fits(rect: tuple[int, int, int, int], used: list[tuple[int, int, int, int]], pad: int = PLACEMENT_PAD) -> bool:
    r0, c0, r1, c1 = rect
    for ur0, uc0, ur1, uc1 in used:
        if not (r1 + pad < ur0 or ur1 + pad < r0 or c1 + pad < uc0 or uc1 + pad < c0):
            return False
    return True


def _place_rect(rows: int, cols: int, h: int, w: int, used: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    positions = []
    for r0 in range(rows - h + 1):
        for c0 in range(cols - w + 1):
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _fits(rect, used):
                positions.append((r0, c0))
    if not positions:
        return None
    return random.choice(positions)


def _pair_count() -> int:
    return random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]


def _build_pairs(pair_count: int) -> list[dict] | None:
    pairs = []
    seen_shapes = set()
    for _ in range(pair_count):
        shape = _random_connected_shape()
        if shape is None:
            return None
        sig = _canonical_shape_signature(shape)
        if sig in seen_shapes:
            return None
        seen_shapes.add(sig)
        pairs.append(
            {
                "shape": shape,
                "template_patch": _make_template_patch(shape),
                "rotation": random.randint(0, 3),
            }
        )
    if len({item["rotation"] for item in pairs}) == 1 and len(pairs) > 1:
        pairs[random.randrange(len(pairs))]["rotation"] = (pairs[0]["rotation"] + 1) % 4
    return pairs


def _assemble_scene(rows: int, cols: int, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((rows, cols), dtype=int)
    expected = np.zeros((rows, cols), dtype=int)
    used: list[tuple[int, int, int, int]] = []
    template_specs = []
    ring_specs = []

    objects = []
    for pair_idx, pair in enumerate(pairs):
        rotated_hole = np.rot90(pair["shape"], pair["rotation"])
        ring_patch = _make_ring_patch(rotated_hole)
        objects.append(("ring", pair_idx, ring_patch))
        objects.append(("template", pair_idx, pair["template_patch"]))
        pair["rotated_hole"] = rotated_hole
        pair["ring_patch"] = ring_patch

    random.shuffle(objects)
    objects.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)

    for kind, pair_idx, patch in objects:
        pos = _place_rect(rows, cols, patch.shape[0], patch.shape[1], used)
        if pos is None:
            return None
        r0, c0 = pos
        paste(inp, patch, r0, c0, transparent=BG, overwrite=True)
        used.append((r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1))
        if kind == "template":
            template_specs.append((pair_idx, r0, c0))
        else:
            ring_specs.append((pair_idx, r0, c0))

    expected[:] = inp
    expected[(expected != BG) & (expected != RING)] = BG

    ring_by_pair = {pair_idx: (r0, c0) for pair_idx, r0, c0 in ring_specs}
    for pair_idx, r0, c0 in template_specs:
        if pair_idx not in ring_by_pair:
            return None
        ring_r0, ring_c0 = ring_by_pair[pair_idx]
        rotated_patch = np.rot90(pairs[pair_idx]["template_patch"], pairs[pair_idx]["rotation"])
        paste(expected, rotated_patch, ring_r0 + 1, ring_c0 + 1, transparent=BG, overwrite=True)

    return inp, expected


def _scene_metadata_ok(inp: np.ndarray, expected: np.ndarray, pair_count: int) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if get_bg(inp) != BG:
        return False

    holes = _collect_holes(inp)
    templates = _collect_templates(inp)
    if len(holes) != pair_count or len(templates) != pair_count:
        return False

    signatures = {_canonical_shape_signature(hole["mask"]) for hole in holes}
    if len(signatures) != pair_count:
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    pair_count = _pair_count()
    pairs = _build_pairs(pair_count)
    if pairs is None:
        return None
    assembled = _assemble_scene(rows, cols, pairs)
    if assembled is None:
        return None
    inp, expected = assembled
    if not _scene_metadata_ok(inp, expected, pair_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-transplant ARC input")

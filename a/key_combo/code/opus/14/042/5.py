import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components, get_bg, label_mask, neighbors8, paste, scale_sprite


BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
BLACK = int(Color.BLACK)

GEN_MAX_RETRIES = 400


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _touches_local_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _local_holes(component_mask: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, list[np.ndarray]]:
    box = bbox(component_mask)
    assert box is not None
    local = crop_bbox(component_mask.astype(int), box).astype(bool)
    bg_labels, n_bg = label_mask(~local, connectivity=4)
    holes = []
    for comp_id in range(1, n_bg + 1):
        hole = bg_labels == comp_id
        if not _touches_local_border(hole):
            holes.append(hole)
    return box, local, holes


def _colored_scaled_patch(local_blue: np.ndarray) -> np.ndarray:
    scaled = scale_sprite(local_blue.astype(int), 2).astype(int)
    padded = np.pad(scaled, 1, constant_values=BLACK)
    black_labels, n_black = label_mask(padded == BLACK, connectivity=4)

    external_ids = set(int(v) for v in np.unique(np.concatenate(
        [
            black_labels[0, :],
            black_labels[-1, :],
            black_labels[:, 0],
            black_labels[:, -1],
        ]
    )))
    external_ids.discard(0)

    colored = padded.copy()
    for r, c in np.argwhere(padded == BLUE):
        sees_external = False
        sees_hole = False
        for nr, nc in neighbors8(int(r), int(c)):
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and padded[nr, nc] == BLACK:
                if int(black_labels[nr, nc]) in external_ids:
                    sees_external = True
                else:
                    sees_hole = True
        if sees_external:
            colored[r, c] = RED
        elif sees_hole:
            colored[r, c] = GREEN

    return colored[1:-1, 1:-1]


def _component_output_patch(component_mask: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    box, local, holes = _local_holes(component_mask)
    if not holes:
        return None
    return box, _colored_scaled_patch(local)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge only blue components that contain holes, then color their outer boundary red and hole-side boundary green.

    Input structure:
    - The background is black and all foreground pixels are blue.
    - Blue pixels form several pairwise separated 4-connected components.
    - Some blue components contain one or more enclosed black holes, while the rest are solid and hole-free.

    Rule:
    - Find every blue 4-connected component.
    - If a component has no enclosed black hole, keep it unchanged.
    - If a component has at least one enclosed black hole, crop it to its tight bounding box, scale that crop by 2 using pixel replication, and place the scaled result back with the same top-left anchor.
    - After scaling, recolor blue pixels on the scaled component: pixels 8-adjacent to the external black region become red, pixels 8-adjacent to hole black cells become green, and red takes priority when both apply.

    Key insight:
    - Hole detection is the shared trigger: the same enclosed-hole test decides both which components are scaled and which scaled boundaries need inside/outside recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    assert bg == BLACK

    out = np.full_like(g, BLACK)
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )

    for comp in components:
        comp_mask = _coords_mask(g.shape, comp["coords"])
        transformed = _component_output_patch(comp_mask)
        if transformed is None:
            out[comp_mask] = BLUE
            continue

        (r0, c0, _r1, _c1), patch = transformed
        assert r0 + patch.shape[0] <= g.shape[0] and c0 + patch.shape[1] <= g.shape[1]
        paste(out, patch, r0, c0, transparent=BLACK, clip=False)

    return out


def _is_connected_blue(patch: np.ndarray) -> bool:
    labels, n = label_mask(patch == BLUE, connectivity=4)
    return n == 1


def _patch_hole_count(patch: np.ndarray) -> int:
    labels, n = label_mask(patch == BLUE, connectivity=4)
    assert n == 1
    _, _, holes = _local_holes(labels == 1)
    return len(holes)


def _boundary_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                cells.append((r, c))
    return cells


def _make_holey_patch(rng: random.Random, requested_holes: int) -> np.ndarray:
    for _ in range(300):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        if requested_holes == 2 and min(h, w) < 5:
            continue

        patch = np.full((h, w), BLUE, dtype=int)
        sizes = [(1, 1), (1, 2), (2, 1), (2, 2)]
        placed_holes: list[tuple[int, int, int, int]] = []

        if requested_holes == 2:
            rng.shuffle(sizes)
            size_choices = None
            for i, first in enumerate(sizes):
                for second in sizes[i + 1:]:
                    if first[0] * first[1] != second[0] * second[1]:
                        size_choices = [first, second]
                        break
                if size_choices is not None:
                    break
            if size_choices is None:
                continue
        else:
            size_choices = [rng.choice(sizes)]

        ok = True
        for hr, hc in size_choices:
            candidates = []
            for r0 in range(1, h - hr):
                for c0 in range(1, w - hc):
                    box = (r0, c0, r0 + hr - 1, c0 + hc - 1)
                    separated = True
                    for pr0, pc0, pr1, pc1 in placed_holes:
                        if not (
                            box[2] + 1 < pr0
                            or pr1 + 1 < box[0]
                            or box[3] + 1 < pc0
                            or pc1 + 1 < box[1]
                        ):
                            separated = False
                            break
                    if separated:
                        candidates.append(box)
            if not candidates:
                ok = False
                break
            box = rng.choice(candidates)
            placed_holes.append(box)
            patch[box[0]:box[2] + 1, box[1]:box[3] + 1] = BLACK
        if not ok:
            continue

        for _ in range(rng.randint(0, 3)):
            boundary = _boundary_cells(h, w)
            rng.shuffle(boundary)
            changed = False
            for r, c in boundary:
                if patch[r, c] != BLUE:
                    continue
                trial = patch.copy()
                trial[r, c] = BLACK
                if _is_connected_blue(trial) and _patch_hole_count(trial) == requested_holes:
                    patch = trial
                    changed = True
                    break
            if not changed:
                break

        if not _is_connected_blue(patch):
            continue
        if _patch_hole_count(patch) != requested_holes:
            continue

        colored = _colored_scaled_patch(patch == BLUE)
        colors = set(int(v) for v in np.unique(colored))
        if RED not in colors or GREEN not in colors:
            continue
        return patch

    raise RuntimeError("failed to construct a holey patch")


def _grow_connected_cells(
    rng: random.Random,
    h: int,
    w: int,
    target_cells: int,
) -> np.ndarray:
    for _ in range(300):
        patch = np.zeros((h, w), dtype=int)
        cells = {(rng.randrange(h), rng.randrange(w))}
        patch[next(iter(cells))] = BLUE
        while len(cells) < target_cells:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            choice = rng.choice(frontier)
            cells.add(choice)
            patch[choice] = BLUE

        if len(cells) != target_cells:
            continue
        cropped_box = bbox(patch == BLUE)
        assert cropped_box is not None
        cropped = crop_bbox(patch, cropped_box)
        if _is_connected_blue(cropped) and _patch_hole_count(cropped) == 0:
            return cropped

    raise RuntimeError("failed to grow a solid patch")


def _make_solid_patch(rng: random.Random, spread: bool) -> np.ndarray:
    for _ in range(300):
        if spread:
            h = rng.randint(4, 5)
            w = rng.randint(4, 5)
            cells = rng.randint(5, 8)
        else:
            h = rng.randint(2, 5)
            w = rng.randint(2, 5)
            cells = rng.randint(3, 8)

        try:
            patch = _grow_connected_cells(rng, h, w, cells)
        except RuntimeError:
            continue
        if spread:
            box = bbox(patch == BLUE)
            assert box is not None
            ph = box[2] - box[0] + 1
            pw = box[3] - box[1] + 1
            if ph < 4 or pw < 4:
                continue
        return patch

    raise RuntimeError("failed to construct a solid patch")


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _compose_expected(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    expected = np.full(shape, BLACK, dtype=int)
    for entry in placements:
        r0, c0 = entry["top_left"]
        if entry["kind"] == "holey":
            patch = _colored_scaled_patch(entry["patch"] == BLUE)
        else:
            patch = entry["patch"]
        paste(expected, patch, r0, c0, transparent=BLACK, clip=False)
    return expected


def _validate_layout(grid: np.ndarray, placements: list[dict]) -> bool:
    blue_components = find_components(
        grid,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    if len(blue_components) != len(placements):
        return False
    if not (3 <= len(placements) <= 5):
        return False

    holey = [entry for entry in placements if entry["kind"] == "holey"]
    solids = [entry for entry in placements if entry["kind"] == "solid"]
    if not (1 <= len(holey) <= 2) or not solids:
        return False

    for entry in holey:
        if _patch_hole_count(entry["patch"]) != entry["holes"]:
            return False
        colored = _colored_scaled_patch(entry["patch"] == BLUE)
        colors = set(int(v) for v in np.unique(colored))
        if RED not in colors or GREEN not in colors:
            return False

    if max((entry["patch"].shape[0] * entry["patch"].shape[1] for entry in solids), default=0) < 16:
        return False

    expected = _compose_expected(grid.shape, placements)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    if not np.any(actual == RED) or not np.any(actual == GREEN):
        return False

    return True


def _place_entries(rng: random.Random, shape: tuple[int, int], entries: list[dict]) -> list[dict] | None:
    h, w = shape
    placed: list[dict] = []
    for entry in sorted(entries, key=lambda item: (item["out_shape"][0] * item["out_shape"][1], item["kind"] == "solid"), reverse=True):
        oh, ow = entry["out_shape"]
        ih, iw = entry["patch"].shape
        for _ in range(600):
            r0 = rng.randint(0, h - oh)
            c0 = rng.randint(0, w - ow)
            in_box = (r0, c0, r0 + ih - 1, c0 + iw - 1)
            out_box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            if any(_rect_overlap(in_box, other["in_box"], pad=2) for other in placed):
                continue
            if any(_rect_overlap(out_box, other["out_box"], pad=1) for other in placed):
                continue
            entry = dict(entry)
            entry["top_left"] = (r0, c0)
            entry["in_box"] = in_box
            entry["out_box"] = out_box
            placed.append(entry)
            break
        else:
            return None
    return placed


def _build_grid(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for entry in placements:
        r0, c0 = entry["top_left"]
        paste(grid, entry["patch"], r0, c0, transparent=BLACK, clip=False)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(22, 28), rng.randint(22, 28))
    total = rng.randint(3, 5)
    holey_count = rng.randint(1, min(2, total - 1))
    solid_count = total - holey_count

    if holey_count == 1:
        hole_plan = [1 if rng.random() < 0.5 else 2]
    else:
        patterns = ([1, 2], [1, 1], [2, 2])
        hole_plan = list(rng.choice(patterns))
        rng.shuffle(hole_plan)

    entries: list[dict] = []
    try:
        for hole_count in hole_plan:
            patch = _make_holey_patch(rng, hole_count)
            ph, pw = patch.shape
            entries.append(
                {
                    "kind": "holey",
                    "holes": hole_count,
                    "patch": patch,
                    "out_shape": (ph * 2, pw * 2),
                }
            )

        for idx in range(solid_count):
            patch = _make_solid_patch(rng, spread=(idx == 0))
            ph, pw = patch.shape
            entries.append(
                {
                    "kind": "solid",
                    "holes": 0,
                    "patch": patch,
                    "out_shape": (ph, pw),
                }
            )
    except RuntimeError:
        return None

    placed = _place_entries(rng, shape, entries)
    if placed is None:
        return None

    grid = _build_grid(shape, placed)
    if not _validate_layout(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

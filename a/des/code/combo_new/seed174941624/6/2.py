from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PIECE_COLORS = [1, 2, 3, 4, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 120
GEN_PLACE_ATTEMPTS = 250


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _rotation_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _trim_mask(mask.astype(bool))
    return min(_mask_key(_trim_mask(np.rot90(base, k))) for k in range(4))


def _has_internal_hole(mask: np.ndarray) -> bool:
    bg = ~mask.astype(bool)
    if not bg.any():
        return False
    for comp in _components(bg):
        if not _touches_border(comp, mask.shape):
            return True
    return False


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors(rr, cc, h, w):
            if not mask[nr, nc]:
                cells.append((rr, cc))
                break
        else:
            if rr in (0, h - 1) or cc in (0, w - 1):
                cells.append((rr, cc))
    return cells


def _piece_patch(mask: np.ndarray, marker_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    patch[marker_mask] = ORANGE
    return patch


def _hole_patch_from_target(target_patch: np.ndarray) -> np.ndarray:
    patch = np.full(target_patch.shape, GREY, dtype=int)
    occupied = target_patch != BLACK
    patch[occupied] = BLACK
    patch[target_patch == ORANGE] = ORANGE
    return patch


def _extract_holes(grid: np.ndarray) -> list[dict[str, object]]:
    holes: list[dict[str, object]] = []
    for comp in _components(grid != GREY):
        if _touches_border(comp, grid.shape):
            continue
        colors = {int(grid[r, c]) for r, c in comp}
        assert colors <= {BLACK, ORANGE}, "template holes should contain only black cells and one orange marker"
        assert ORANGE in colors, "each hole must contain one orange marker"

        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        marker = np.zeros_like(mask)
        for r, c in comp:
            mask[r - r0, c - c0] = True
            if int(grid[r, c]) == ORANGE:
                marker[r - r0, c - c0] = True

        assert int(marker.sum()) == 1, "each hole must have exactly one orange marker"
        holes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "marker": marker,
            }
        )

    assert holes, "expected at least one enclosed hole"
    return holes


def _extract_fragments(grid: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != BLACK) & (grid != GREY)
    fragments: list[dict[str, object]] = []
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        if colors == {ORANGE}:
            continue
        assert ORANGE in colors, "each fragment must contain an orange marker"
        assert len(colors - {ORANGE}) == 1, "each fragment must be monochrome apart from its marker"

        r0, c0, r1, c1 = _bbox(comp)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
        marker = np.zeros_like(patch, dtype=bool)
        for r, c in comp:
            patch[r - r0, c - c0] = int(grid[r, c])
            if int(grid[r, c]) == ORANGE:
                marker[r - r0, c - c0] = True

        assert int(marker.sum()) == 1, "each fragment must have exactly one orange marker"
        fragments.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": comp,
                "patch": patch,
            }
        )

    assert fragments, "expected at least one external fragment"
    return fragments


def _match_fragment_to_hole(fragment: dict[str, object], hole: dict[str, object]) -> np.ndarray | None:
    piece_patch = np.asarray(fragment["patch"], dtype=int)
    hole_mask = np.asarray(hole["mask"], dtype=bool)
    hole_marker = np.asarray(hole["marker"], dtype=bool)

    matches: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(piece_patch, k)
        if rotated.shape != hole_mask.shape:
            continue
        if not np.array_equal(rotated != BLACK, hole_mask):
            continue
        if not np.array_equal(rotated == ORANGE, hole_marker):
            continue
        matches.append(rotated)

    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered fragments into the marked holes of a grey template and erase their original copies.

    Input structure:
    - The background is black.
    - One large connected grey template sits near the middle of the grid.
    - Inside the template are one or more enclosed holes made of black cells, and each hole contains exactly one orange marker cell on its edge.
    - Outside the template are several small connected monochrome fragments; each fragment uses one non-black, non-grey color plus exactly one orange marker cell on its edge.

    Rule:
    - Find every enclosed non-grey region inside the template; each such region is one hole shape.
    - Match each hole with the unique outside fragment whose shape can be rotated to occupy the same cells and whose orange marker lands on the hole's orange marker.
    - Fill the hole with that rotated fragment, preserving the orange marker cell, and clear the fragment from its original outside position back to black.

    Key insight:
    - Grey acts only as the template wall. The actual matching object for a hole is the enclosed non-grey region itself, while the orange marker removes any rotational ambiguity that shape matching alone would leave.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    holes = _extract_holes(g)
    fragments = _extract_fragments(g)
    assert len(holes) == len(fragments), "holes and fragments must match one-to-one"

    assignments: list[tuple[dict[str, object], np.ndarray]] = []
    used_fragments: set[int] = set()
    for hole in holes:
        candidates: list[tuple[int, np.ndarray]] = []
        for idx, fragment in enumerate(fragments):
            placed = _match_fragment_to_hole(fragment, hole)
            if placed is not None:
                candidates.append((idx, placed))
        assert len(candidates) == 1, "each hole must match exactly one fragment and rotation"
        frag_idx, placed_patch = candidates[0]
        assert frag_idx not in used_fragments, "a fragment cannot fill multiple holes"
        used_fragments.add(frag_idx)
        assignments.append((hole, placed_patch))

    out = g.copy()
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = BLACK

    for hole, placed_patch in assignments:
        r0, c0, r1, c1 = hole["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = np.asarray(hole["mask"], dtype=bool)
        region[mask] = placed_patch[mask]

    return out


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        box_h = rng.randint(2, 5)
        box_w = rng.randint(2, 5)
        max_area = min(8, box_h * box_w)
        area = rng.randint(4, max_area)

        start = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors(r, c, box_h, box_w):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if len(_components(mask)) != 1:
            continue
        if _has_internal_hole(mask):
            continue
        if len(_boundary_cells(mask)) < 2:
            continue
        return mask
    return None


def _count_rotational_matches(piece_patch: np.ndarray, hole_patch: np.ndarray) -> int:
    hole_mask = hole_patch != GREY
    hole_marker = hole_patch == ORANGE
    count = 0
    for k in range(4):
        rotated = np.rot90(piece_patch, k)
        if rotated.shape != hole_patch.shape:
            continue
        if np.array_equal(rotated != BLACK, hole_mask) and np.array_equal(rotated == ORANGE, hole_marker):
            count += 1
    return count


def _build_instances(rng: random.Random, count: int) -> list[dict[str, object]] | None:
    signatures: set[tuple[str, ...]] = set()
    colors = rng.sample(PIECE_COLORS, count)
    instances: list[dict[str, object]] = []

    for color in colors:
        mask = _random_shape(rng)
        if mask is None:
            return None
        sig = _rotation_signature(mask)
        if sig in signatures:
            return None
        signatures.add(sig)

        boundary = _boundary_cells(mask)
        rng.shuffle(boundary)
        built = None
        for marker_r, marker_c in boundary:
            marker_mask = np.zeros_like(mask, dtype=bool)
            marker_mask[marker_r, marker_c] = True
            base_piece = _piece_patch(mask, marker_mask, color)

            piece_rot = rng.randrange(4)
            hole_rot = rng.randrange(4)
            input_piece = np.rot90(base_piece, piece_rot)
            target_piece = np.rot90(base_piece, hole_rot)
            hole_patch = _hole_patch_from_target(target_piece)
            needed_rotation = (hole_rot - piece_rot) % 4

            if not np.array_equal(np.rot90(input_piece, needed_rotation), target_piece):
                continue
            if _count_rotational_matches(input_piece, hole_patch) != 1:
                continue

            built = {
                "input_piece": input_piece,
                "target_piece": target_piece,
                "hole_patch": hole_patch,
            }
            break

        if built is None:
            return None
        instances.append(built)

    return instances


def _layout_holes(rng: random.Random, hole_patches: list[np.ndarray]) -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    count = len(hole_patches)
    order = list(range(count))
    rng.shuffle(order)

    if count == 1:
        rows = [order]
    elif count == 2:
        rows = [order] if rng.random() < 0.5 else [[order[0]], [order[1]]]
    else:
        if rng.random() < 0.5:
            rows = [order]
        else:
            split = rng.randint(1, count - 1)
            rows = [order[:split], order[split:]]

    row_specs: list[dict[str, object]] = []
    base_inner_h = 0
    base_inner_w = 0
    row_gaps: list[int] = []
    for row in rows:
        widths = [hole_patches[idx].shape[1] for idx in row]
        heights = [hole_patches[idx].shape[0] for idx in row]
        gaps = [rng.randint(1, 2) for _ in range(max(0, len(row) - 1))]
        row_width = sum(widths) + sum(gaps)
        row_height = max(heights) + rng.randint(0, 1)
        row_specs.append(
            {
                "indices": row,
                "gaps": gaps,
                "row_width": row_width,
                "row_height": row_height,
            }
        )
        base_inner_w = max(base_inner_w, row_width)
        base_inner_h += row_height

    for _ in range(max(0, len(rows) - 1)):
        gap = rng.randint(1, 2)
        row_gaps.append(gap)
        base_inner_h += gap

    inner_w = base_inner_w + rng.randint(0, 2)
    inner_h = base_inner_h + rng.randint(0, 2)
    template_h = inner_h + 2
    template_w = inner_w + 2
    if template_h > 20 or template_w > 20:
        return None

    top_slack = inner_h - base_inner_h
    current_y = rng.randint(0, top_slack) if top_slack else 0
    positions: dict[int, tuple[int, int]] = {}

    for row_idx, spec in enumerate(row_specs):
        row = spec["indices"]
        row_width = int(spec["row_width"])
        row_height = int(spec["row_height"])
        start_x = rng.randint(0, inner_w - row_width) if inner_w > row_width else 0
        cursor_x = start_x

        for hole_idx, obj_idx in enumerate(row):
            patch = hole_patches[obj_idx]
            slack_y = row_height - patch.shape[0]
            offset_y = rng.randint(0, slack_y) if slack_y else 0
            positions[obj_idx] = (1 + current_y + offset_y, 1 + cursor_x)
            cursor_x += patch.shape[1]
            if hole_idx < len(row) - 1:
                cursor_x += int(spec["gaps"][hole_idx])

        current_y += row_height
        if row_idx < len(row_gaps):
            current_y += row_gaps[row_idx]

    return template_h, template_w, positions


def _reserve_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    view = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _generate_one(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(2, 4)
    instances = _build_instances(rng, piece_count)
    if instances is None:
        return None

    hole_patches = [np.asarray(instance["hole_patch"], dtype=int) for instance in instances]
    layout = _layout_holes(rng, hole_patches)
    if layout is None:
        return None
    template_h, template_w, hole_positions = layout

    extra_h = rng.randint(8, 12)
    extra_w = rng.randint(8, 12)
    grid_h = template_h + extra_h
    grid_w = template_w + extra_w
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    template_r0 = max(0, min(grid_h - template_h, (grid_h - template_h) // 2 + rng.randint(-1, 1)))
    template_c0 = max(0, min(grid_w - template_w, (grid_w - template_w) // 2 + rng.randint(-1, 1)))
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = GREY

    expected = grid.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve_box(occupied, template_r0, template_c0, template_h, template_w, pad=1)

    for idx, instance in enumerate(instances):
        hole_patch = np.asarray(instance["hole_patch"], dtype=int)
        target_piece = np.asarray(instance["target_piece"], dtype=int)
        hr, hc = hole_positions[idx]
        gr = template_r0 + hr
        gc = template_c0 + hc

        grid[gr : gr + hole_patch.shape[0], gc : gc + hole_patch.shape[1]] = hole_patch
        mask = target_piece != BLACK
        expected[gr : gr + target_piece.shape[0], gc : gc + target_piece.shape[1]][mask] = target_piece[mask]

    placements: list[tuple[np.ndarray, int, int]] = []
    for instance in instances:
        piece_patch = np.asarray(instance["input_piece"], dtype=int)
        ph, pw = piece_patch.shape
        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            pr = rng.randint(0, grid_h - ph)
            pc = rng.randint(0, grid_w - pw)
            if not _can_place_box(occupied, pr, pc, ph, pw, pad=1):
                continue
            placements.append((piece_patch, pr, pc))
            _reserve_box(occupied, pr, pc, ph, pw, pad=1)
            placed = True
            break
        if not placed:
            return None

    for piece_patch, pr, pc in placements:
        _paint_patch(grid, piece_patch, pr, pc)
        mask = piece_patch != BLACK
        expected[pr : pr + piece_patch.shape[0], pc : pc + piece_patch.shape[1]][mask] = BLACK

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _generate_one(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

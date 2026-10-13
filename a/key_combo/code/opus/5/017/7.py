from __future__ import annotations

from collections import Counter
import random

import numpy as np


GRID_SIZE = 16
CENTER_R0 = 4
CENTER_C0 = 4
CENTER_SIZE = 8
QUAD_SIZE = 4
BG = 0
GRAY = 5
MARKER_COLORS = (1, 2, 3, 4)
MAX_RETRIES = 500


def _coords_where(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def _infer_axis_scores(mask: np.ndarray, axis: str) -> list[tuple[int, int]]:
    """Return (doubled_axis_position, symmetry_pair_count) candidates."""
    assert mask.shape == (CENTER_SIZE, CENTER_SIZE)
    coords = _coords_where(mask)
    scores: list[tuple[int, int]] = []
    for a2 in range(0, 2 * (CENTER_SIZE - 1) + 1):
        pairs = 0
        for r, c in coords:
            r = int(r)
            c = int(c)
            if axis == "vertical":
                c2 = int(a2 - c)
                if 0 <= c2 < CENTER_SIZE and c <= c2 and bool(mask[r, c2]):
                    pairs += 1
            else:
                r2 = int(a2 - r)
                if 0 <= r2 < CENTER_SIZE and r <= r2 and bool(mask[r2, c]):
                    pairs += 1
        scores.append((int(a2), int(pairs)))
    return scores


def _infer_axis(mask: np.ndarray, axis: str) -> int:
    scores = _infer_axis_scores(mask, axis)
    return max(scores, key=lambda item: item[1])[0]


def _repair_by_symmetry(region: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Fill missing cells in the 8x8 center from inferred horizontal and vertical mirrors."""
    assert region.shape == (CENTER_SIZE, CENTER_SIZE)
    visible = region == GRAY
    a2 = _infer_axis(visible, "vertical")
    b2 = _infer_axis(visible, "horizontal")

    out = region.copy()
    for _ in range(CENTER_SIZE):
        changed = False
        for r in range(CENTER_SIZE):
            for c in range(CENTER_SIZE):
                if int(out[r, c]) != BG:
                    continue
                candidates: list[int] = []

                c2 = int(a2 - c)
                if 0 <= c2 < CENTER_SIZE and int(out[r, c2]) != BG:
                    candidates.append(int(out[r, c2]))

                r2 = int(b2 - r)
                if 0 <= r2 < CENTER_SIZE and int(out[r2, c]) != BG:
                    candidates.append(int(out[r2, c]))

                if 0 <= r2 < CENTER_SIZE and 0 <= c2 < CENTER_SIZE and int(out[r2, c2]) != BG:
                    candidates.append(int(out[r2, c2]))

                if candidates:
                    out[r, c] = int(Counter(candidates).most_common(1)[0][0])
                    changed = True
        if not changed:
            break
    return out, int(a2), int(b2)


def _quadrants(region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        region[:QUAD_SIZE, :QUAD_SIZE].copy(),
        region[:QUAD_SIZE, QUAD_SIZE:].copy(),
        region[QUAD_SIZE:, :QUAD_SIZE].copy(),
        region[QUAD_SIZE:, QUAD_SIZE:].copy(),
    )


def _stamp_template(template: np.ndarray, corner: str) -> np.ndarray:
    if corner == "top_left":
        return template[::-1, ::-1].copy()
    if corner == "top_right":
        return template[::-1, :].copy()
    if corner == "bottom_left":
        return template[:, ::-1].copy()
    if corner == "bottom_right":
        return template.copy()
    raise ValueError(f"unknown corner {corner}")


def _render_output_from_repaired(inp: np.ndarray, repaired_center: np.ndarray) -> np.ndarray:
    out = np.array(inp, dtype=int).copy()
    out[CENTER_R0 : CENTER_R0 + CENTER_SIZE, CENTER_C0 : CENTER_C0 + CENTER_SIZE] = repaired_center

    quadrants = {
        "top_left": repaired_center[:QUAD_SIZE, :QUAD_SIZE],
        "top_right": repaired_center[:QUAD_SIZE, QUAD_SIZE:],
        "bottom_left": repaired_center[QUAD_SIZE:, :QUAD_SIZE],
        "bottom_right": repaired_center[QUAD_SIZE:, QUAD_SIZE:],
    }
    corners = {
        "top_left": (0, 0, 0, 0),
        "top_right": (0, QUAD_SIZE, 0, GRID_SIZE - 1 - (QUAD_SIZE - 1)),
        "bottom_left": (QUAD_SIZE, 0, GRID_SIZE - 1 - (QUAD_SIZE - 1), 0),
        "bottom_right": (QUAD_SIZE, QUAD_SIZE, GRID_SIZE - 1 - (QUAD_SIZE - 1), GRID_SIZE - 1 - (QUAD_SIZE - 1)),
    }
    marker_positions = {
        "top_left": (0, 0),
        "top_right": (0, GRID_SIZE - 1),
        "bottom_left": (GRID_SIZE - 1, 0),
        "bottom_right": (GRID_SIZE - 1, GRID_SIZE - 1),
    }

    for corner, (mr, mc) in marker_positions.items():
        color = int(inp[mr, mc])
        out[mr, mc] = BG
        stamped = _stamp_template(quadrants[corner], corner)
        if corner == "top_left":
            block = (slice(0, QUAD_SIZE), slice(0, QUAD_SIZE))
        elif corner == "top_right":
            block = (slice(0, QUAD_SIZE), slice(GRID_SIZE - QUAD_SIZE, GRID_SIZE))
        elif corner == "bottom_left":
            block = (slice(GRID_SIZE - QUAD_SIZE, GRID_SIZE), slice(0, QUAD_SIZE))
        else:
            block = (slice(GRID_SIZE - QUAD_SIZE, GRID_SIZE), slice(GRID_SIZE - QUAD_SIZE, GRID_SIZE))
        mask = stamped == GRAY
        out[block][mask] = color
    return out


def _build_full_center(base_mask: np.ndarray) -> np.ndarray:
    ul = base_mask.astype(int) * GRAY
    ur = ul[:, ::-1]
    ll = ul[::-1, :]
    lr = ul[::-1, ::-1]
    return np.block([[ul, ur], [ll, lr]]).astype(int)


def _base_to_orbit_coords(r: int, c: int) -> dict[str, tuple[int, int]]:
    return {
        "UL": (r, c),
        "UR": (r, CENTER_SIZE - 1 - c),
        "LL": (CENTER_SIZE - 1 - r, c),
        "LR": (CENTER_SIZE - 1 - r, CENTER_SIZE - 1 - c),
    }


def _hole_branch_tags(center_before: np.ndarray, missing_cells: set[tuple[int, int]], a2: int, b2: int) -> set[str]:
    tags: set[str] = set()
    for r, c in missing_cells:
        c2 = int(a2 - c)
        r2 = int(b2 - r)
        v = 0 <= c2 < CENTER_SIZE and int(center_before[r, c2]) == GRAY
        h = 0 <= r2 < CENTER_SIZE and int(center_before[r2, c]) == GRAY
        d = 0 <= r2 < CENTER_SIZE and 0 <= c2 < CENTER_SIZE and int(center_before[r2, c2]) == GRAY
        if d and not v and not h:
            tags.add("double_only")
        elif v and not h:
            tags.add("vertical_available")
        elif h and not v:
            tags.add("horizontal_available")
        elif v and h:
            tags.add("both_available")
    return tags


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the erased symmetric center, extract its quadrant templates, and stamp them into the corners using the marker colors.

    Input structure:
    - The grid is 16x16 with black background.
    - The center 8x8 window at rows 4-11 and cols 4-11 contains a gray mosaic that is symmetric across one horizontal and one vertical reflection axis, but some gray cells are erased to black.
    - The four outer corners contain distinct non-black marker pixels whose colors choose the stamp colors.

    Rule:
    - Infer the center mosaic's vertical and horizontal reflection axes from the visible gray cells by choosing the axes with the strongest surviving symmetry.
    - Fill each black center cell from its vertical mirror, horizontal mirror, or double mirror whenever one of those mirrored cells is non-black.
    - Split the repaired 8x8 center into four 4x4 quadrants, treat each quadrant's gray-cell pattern as a template, erase the corner markers, and stamp the corresponding template into that corner in the marker's color with the template's inward-facing corner aligned to the marker cell.
    - Keep the repaired gray center and leave all other cells black.

    Key insight:
    - The corner templates are not given directly; they only become available after reconstructing the hidden parts of the doubly symmetric center pattern.
    """
    g = np.array(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE)

    center = g[CENTER_R0 : CENTER_R0 + CENTER_SIZE, CENTER_C0 : CENTER_C0 + CENTER_SIZE].copy()
    repaired_center, _a2, _b2 = _repair_by_symmetry(center)
    return _render_output_from_repaired(g, repaired_center)


def _random_base_mask() -> np.ndarray | None:
    cells = [(r, c) for r in range(QUAD_SIZE) for c in range(QUAD_SIZE)]
    random.shuffle(cells)
    count = random.randint(6, 10)
    chosen = set(cells[:count])
    mask = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True

    if np.array_equal(mask, mask[::-1, :]):
        return None
    if np.array_equal(mask, mask[:, ::-1]):
        return None
    if min(mask.sum(axis=0)) == 0 or min(mask.sum(axis=1)) == 0:
        return None
    if len({tuple(row.tolist()) for row in mask}) < 2:
        return None
    return mask


def _choose_hole_scheme(occupied_base_cells: list[tuple[int, int]]) -> list[tuple[str, ...]] | None:
    schemes = [
        [("UL", "UR", "LL"), ("LR",)],
        [("UL", "UR"), ("UL", "LL"), ("LR",)],
        [("UL", "UR", "LL"), ("LR",), ("UR",)],
        [("UL", "UR"), ("UL", "LL"), ("LR",), ("UR",)],
        [("UL", "UR", "LL"), ("LR",), ("UL", "LL")],
    ]
    candidates = [scheme for scheme in schemes if len(occupied_base_cells) >= len(scheme)]
    if not candidates:
        return None
    return random.choice(candidates)


def _apply_holes(full_center: np.ndarray, base_mask: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], list[tuple[str, ...]]] | None:
    occupied = [(int(r), int(c)) for r, c in np.argwhere(base_mask)]
    random.shuffle(occupied)
    scheme = _choose_hole_scheme(occupied)
    if scheme is None:
        return None

    center = full_center.copy()
    missing: set[tuple[int, int]] = set()
    for (r, c), labels in zip(occupied, scheme):
        orbit = _base_to_orbit_coords(r, c)
        for label in labels:
            rr, cc = orbit[label]
            center[rr, cc] = BG
            missing.add((int(rr), int(cc)))
    return center, missing, scheme


def _validate_semantics(inp: np.ndarray, full_center: np.ndarray, visible_center: np.ndarray, missing: set[tuple[int, int]]) -> bool:
    if len(missing) < 4 or len(missing) > 8:
        return False

    quadrants_full = _quadrants(full_center)
    if len({quad.tobytes() for quad in quadrants_full}) < 2:
        return False

    quadrant_slices = [
        (slice(0, QUAD_SIZE), slice(0, QUAD_SIZE)),
        (slice(0, QUAD_SIZE), slice(QUAD_SIZE, CENTER_SIZE)),
        (slice(QUAD_SIZE, CENTER_SIZE), slice(0, QUAD_SIZE)),
        (slice(QUAD_SIZE, CENTER_SIZE), slice(QUAD_SIZE, CENTER_SIZE)),
    ]
    for rs, cs in quadrant_slices:
        if not np.any((full_center[rs, cs] == GRAY) & (visible_center[rs, cs] == BG)):
            return False

    visible_mask = visible_center == GRAY
    v_scores = _infer_axis_scores(visible_mask, "vertical")
    h_scores = _infer_axis_scores(visible_mask, "horizontal")
    best_v = max(score for _axis, score in v_scores)
    best_h = max(score for _axis, score in h_scores)
    if sum(score == best_v for _axis, score in v_scores) != 1:
        return False
    if sum(score == best_h for _axis, score in h_scores) != 1:
        return False
    if _infer_axis(visible_mask, "vertical") != CENTER_SIZE - 1:
        return False
    if _infer_axis(visible_mask, "horizontal") != CENTER_SIZE - 1:
        return False

    tags = _hole_branch_tags(visible_center, missing, CENTER_SIZE - 1, CENTER_SIZE - 1)
    if not ({"vertical_available", "horizontal_available"} <= tags or "double_only" in tags):
        return False

    expected = _render_output_from_repaired(inp, full_center)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    for corner, (r, c) in {
        "top_left": (0, 0),
        "top_right": (0, GRID_SIZE - 1),
        "bottom_left": (GRID_SIZE - 1, 0),
        "bottom_right": (GRID_SIZE - 1, GRID_SIZE - 1),
    }.items():
        if int(inp[r, c]) not in MARKER_COLORS:
            return False
        if int(actual[r, c]) not in (BG, int(inp[r, c])):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    base_mask = _random_base_mask()
    if base_mask is None:
        return None

    full_center = _build_full_center(base_mask)
    applied = _apply_holes(full_center, base_mask)
    if applied is None:
        return None
    visible_center, missing, _scheme = applied

    inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    inp[CENTER_R0 : CENTER_R0 + CENTER_SIZE, CENTER_C0 : CENTER_C0 + CENTER_SIZE] = visible_center

    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    corners = [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]
    for color, (r, c) in zip(colors, corners):
        inp[r, c] = int(color)

    if not _validate_semantics(inp, full_center, visible_center, missing):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")

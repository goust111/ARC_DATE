from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
MARKER_COLORS = (4, 5, 6, 7, 8, 9)

CORNER_TO_ROT = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
OPPOSITE_CORNER = {"TL": "BR", "TR": "BL", "BR": "TL", "BL": "TR"}
MISSING_TO_RECT_DELTA = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}
ROTATED_TL_CORNER = {0: "TL", 1: "BL", 2: "BR", 3: "TR"}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _marker_info(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    cells = set(coords)
    all_cells = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = list(all_cells - cells)
    if len(missing) != 1:
        return None
    mr, mc = missing[0]
    if (mr, mc) == (r0, c0):
        missing_corner = "TL"
    elif (mr, mc) == (r0, c1):
        missing_corner = "TR"
    elif (mr, mc) == (r1, c0):
        missing_corner = "BL"
    else:
        missing_corner = "BR"
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "missing_cell": (mr, mc),
        "missing_corner": missing_corner,
        "anchor_corner": OPPOSITE_CORNER[missing_corner],
    }


def _extract_objects(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    marker_infos: list[dict[str, object]] = []
    marker_mask = np.zeros_like(grid, dtype=bool)
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        for comp in _components(grid == color):
            info = _marker_info(comp)
            if info is not None:
                info["color"] = color
                marker_infos.append(info)
                for r, c in comp:
                    marker_mask[r, c] = True
    if len(marker_infos) != 2:
        raise AssertionError("expected exactly two L markers")

    rect_infos: list[dict[str, object]] = []
    for comp in _components((grid != BLACK) & (~marker_mask)):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(comp) != patch.size:
            raise AssertionError("mask object must be a filled rectangle")
        colors = set(int(v) for v in np.unique(patch))
        if not colors <= {BLUE, RED} or len(colors) != 2:
            raise AssertionError("mask rectangle must use the two binary mask colors")
        corners = {
            "TL": (r0, c0),
            "TR": (r0, c1),
            "BL": (r1, c0),
            "BR": (r1, c1),
        }
        rect_infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch.copy(),
                "corners": corners,
                "point_to_corner": {v: k for k, v in corners.items()},
            }
        )
    if len(rect_infos) != 2:
        raise AssertionError("expected exactly two mask rectangles")
    return rect_infos, marker_infos


def _normalize_patch(rect_info: dict[str, object], marker_info: dict[str, object]) -> np.ndarray:
    mr, mc = marker_info["missing_cell"]  # type: ignore[assignment]
    dr, dc = MISSING_TO_RECT_DELTA[marker_info["missing_corner"]]  # type: ignore[index]
    corner_point = (mr + dr, mc + dc)
    point_to_corner = rect_info["point_to_corner"]  # type: ignore[assignment]
    if corner_point not in point_to_corner:
        raise AssertionError("marker does not point to a rectangle corner")
    anchor_corner = point_to_corner[corner_point]
    if anchor_corner != marker_info["anchor_corner"]:
        raise AssertionError("marker/rectangle corner mismatch")
    patch = rect_info["patch"]  # type: ignore[assignment]
    return np.rot90(patch, CORNER_TO_ROT[anchor_corner])


def _xor_output(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise AssertionError("normalized masks must have the same shape")
    return np.where((a == RED) ^ (b == RED), GREEN, BLACK).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered marker-tagged mask to a shared anchor corner, then XOR them cell by cell.

    Input structure:
    - The background contains exactly two separated filled rectangular mask regions made from blue and red cells.
    - Each rectangle is paired with one external 3-pixel L marker (a 2x2 block missing one corner).
    - The marker gap points to one corner of its rectangle, indicating which rectangle corner is the anchor.

    Rule:
    - Read each L marker to determine the rectangle corner it tags.
    - Rotate each whole rectangle so that its tagged corner becomes the top-left corner.
    - Compare the two normalized masks cell by cell and output green where exactly one mask cell is red, otherwise black.

    Key insight:
    - The rectangles are not aligned by their positions on the canvas; the L markers define the only correct orientation for alignment.
    """
    g = np.asarray(grid, dtype=int)
    rect_infos, marker_infos = _extract_objects(g)

    normalized: list[np.ndarray] = []
    used_rects: set[int] = set()
    for marker in marker_infos:
        mr, mc = marker["missing_cell"]  # type: ignore[assignment]
        dr, dc = MISSING_TO_RECT_DELTA[marker["missing_corner"]]  # type: ignore[index]
        corner_point = (mr + dr, mc + dc)

        matched_idx = None
        for idx, rect in enumerate(rect_infos):
            if corner_point in rect["point_to_corner"] and idx not in used_rects:  # type: ignore[operator]
                matched_idx = idx
                break
        if matched_idx is None:
            raise AssertionError("could not associate marker with a unique rectangle")
        used_rects.add(matched_idx)
        normalized.append(_normalize_patch(rect_infos[matched_idx], marker))

    if len(normalized) != 2:
        raise AssertionError("expected two normalized masks")
    return _xor_output(normalized[0], normalized[1])


def _build_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    for _ in range(200):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(random.randint(2, 5)):
            r0 = random.randint(0, h - 1)
            r1 = random.randint(r0, h - 1)
            c0 = random.randint(0, w - 1)
            c1 = random.randint(c0, w - 1)
            if (r1 - r0 + 1) * (c1 - c0 + 1) >= area:
                continue
            if random.random() < 0.7:
                mask[r0 : r1 + 1, c0 : c1 + 1] = True
            else:
                mask[r0 : r1 + 1, c0 : c1 + 1] ^= True
        for _ in range(random.randint(1, 3)):
            mask[random.randrange(h), random.randrange(w)] ^= True
        red_count = int(mask.sum())
        if red_count < 3 or red_count > area - 3:
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if h == w and any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate binary mask")


def _mask_to_patch(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, RED, BLUE).astype(int)


def _marker_bbox_for_rect(r: int, c: int, h: int, w: int, anchor_corner: str) -> tuple[int, int]:
    if anchor_corner == "TL":
        return r - 2, c - 2
    if anchor_corner == "TR":
        return r - 2, c + w
    if anchor_corner == "BL":
        return r + h, c - 2
    return r + h, c + w


def _marker_cells(r: int, c: int, h: int, w: int, anchor_corner: str) -> list[tuple[int, int]]:
    mr0, mc0 = _marker_bbox_for_rect(r, c, h, w, anchor_corner)
    all_cells = [(mr0, mc0), (mr0, mc0 + 1), (mr0 + 1, mc0), (mr0 + 1, mc0 + 1)]
    missing_corner = OPPOSITE_CORNER[anchor_corner]
    missing = {
        "TL": (mr0, mc0),
        "TR": (mr0, mc0 + 1),
        "BL": (mr0 + 1, mc0),
        "BR": (mr0 + 1, mc0 + 1),
    }[missing_corner]
    return [cell for cell in all_cells if cell != missing]


def _envelope(r: int, c: int, h: int, w: int, anchor_corner: str) -> tuple[int, int, int, int]:
    top = r - (2 if anchor_corner in {"TL", "TR"} else 0)
    left = c - (2 if anchor_corner in {"TL", "BL"} else 0)
    bottom = r + h - 1 + (2 if anchor_corner in {"BL", "BR"} else 0)
    right = c + w - 1 + (2 if anchor_corner in {"TR", "BR"} else 0)
    return top, left, bottom, right


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _sample_position(
    canvas_h: int,
    canvas_w: int,
    patch_h: int,
    patch_w: int,
    anchor_corner: str,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    choices: list[tuple[int, int]] = []
    top_margin = 2 if anchor_corner in {"TL", "TR"} else 0
    left_margin = 2 if anchor_corner in {"TL", "BL"} else 0
    bottom_margin = 2 if anchor_corner in {"BL", "BR"} else 0
    right_margin = 2 if anchor_corner in {"TR", "BR"} else 0
    for r in range(top_margin, canvas_h - patch_h - bottom_margin + 1):
        for c in range(left_margin, canvas_w - patch_w - right_margin + 1):
            env = _envelope(r, c, patch_h, patch_w, anchor_corner)
            if all(_boxes_separated(env, other, gap=1) for other in occupied):
                choices.append((r, c))
    return random.choice(choices) if choices else None


def _alternative_orientation_matches(
    patch1: np.ndarray,
    patch2: np.ndarray,
    rot1: int,
    rot2: int,
    expected: np.ndarray,
) -> bool:
    target_shape = expected.shape
    options1 = [k for k in range(4) if np.rot90(patch1, k).shape == target_shape]
    options2 = [k for k in range(4) if np.rot90(patch2, k).shape == target_shape]
    for k1 in options1:
        for k2 in options2:
            if (k1, k2) == (rot1, rot2):
                continue
            candidate = _xor_output(np.rot90(patch1, k1), np.rot90(patch2, k2))
            if np.array_equal(candidate, expected):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    if random.random() < 0.8:
        h = random.randint(4, 8)
        w_choices = [v for v in range(4, 9) if v != h]
        w = random.choice(w_choices)
    else:
        h = w = random.randint(4, 7)

    mask_a = _build_mask(h, w)
    mask_b = _build_mask(h, w)
    patch_a = _mask_to_patch(mask_a)
    patch_b = _mask_to_patch(mask_b)
    expected = _xor_output(patch_a, patch_b)
    green_count = int((expected == GREEN).sum())
    if green_count < 3 or green_count > h * w - 3:
        return None

    rotations = random.sample(range(4), 2)
    in_a = np.rot90(patch_a, rotations[0])
    in_b = np.rot90(patch_b, rotations[1])
    anchor_a = ROTATED_TL_CORNER[rotations[0]]
    anchor_b = ROTATED_TL_CORNER[rotations[1]]
    rot_back_a = CORNER_TO_ROT[anchor_a]
    rot_back_b = CORNER_TO_ROT[anchor_b]

    if _alternative_orientation_matches(in_a, in_b, rot_back_a, rot_back_b, expected):
        return None

    marker_color = random.choice(MARKER_COLORS)
    patch_specs = [(in_a, anchor_a), (in_b, anchor_b)]

    min_h = max(p.shape[0] for p, _ in patch_specs) + 8
    min_w = max(p.shape[1] for p, _ in patch_specs) + 8
    canvas_h = random.randint(max(14, min_h), 30)
    canvas_w = random.randint(max(14, min_w), 30)
    canvas = np.full((canvas_h, canvas_w), BLACK, dtype=int)

    occupied: list[tuple[int, int, int, int]] = []
    placements: list[tuple[np.ndarray, str, int, int]] = []
    for patch, anchor in random.sample(patch_specs, k=2):
        pos = _sample_position(canvas_h, canvas_w, patch.shape[0], patch.shape[1], anchor, occupied)
        if pos is None:
            return None
        r, c = pos
        env = _envelope(r, c, patch.shape[0], patch.shape[1], anchor)
        occupied.append(env)
        placements.append((patch, anchor, r, c))

    for patch, anchor, r, c in placements:
        canvas[r : r + patch.shape[0], c : c + patch.shape[1]] = patch
        for rr, cc in _marker_cells(r, c, patch.shape[0], patch.shape[1], anchor):
            canvas[rr, cc] = marker_color

    try:
        actual = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    rect_infos, marker_infos = _extract_objects(canvas)
    if len(rect_infos) != 2 or len(marker_infos) != 2:
        return None

    if in_a.shape == in_b.shape:
        naive = _xor_output(in_a, in_b)
        if np.array_equal(naive, expected):
            return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-normalized XOR task")

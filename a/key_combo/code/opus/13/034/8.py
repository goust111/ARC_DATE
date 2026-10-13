from __future__ import annotations

import hashlib
import random
from collections import deque
from itertools import combinations, permutations

import numpy as np


SIDES = ("up", "down", "left", "right")
PALETTE = (1, 2, 4, 7)
OUTPUT_SIDE = 13
CENTER_SIZE = 3
ARM_LEN = 2
INPUT_ATTEMPTS = 400
_SEEN_OUTPUT_SIGNATURES: set[str] = set()
_CONFIG_QUEUE: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = 0) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q: deque[tuple[int, int]] = deque([tuple(map(int, coords[0]))])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                pt = (nr, nc)
                if pt not in seen:
                    seen.add(pt)
                    q.append(pt)
    return len(seen) == int(mask.sum())


def _body_candidates(shape: tuple[int, int]) -> list[tuple[str, tuple[slice, slice], tuple[tuple[int, int], ...], tuple[int, int]]]:
    if shape == (5, 3):
        return [
            ("up", (slice(2, 5), slice(0, 3)), ((0, 1), (1, 1)), (0, 1)),
            ("down", (slice(0, 3), slice(0, 3)), ((3, 1), (4, 1)), (2, 1)),
        ]
    if shape == (3, 5):
        return [
            ("left", (slice(0, 3), slice(2, 5)), ((1, 0), (1, 1)), (1, 0)),
            ("right", (slice(0, 3), slice(0, 3)), ((1, 3), (1, 4)), (1, 2)),
        ]
    return []


def _analyze_sprite(patch: np.ndarray, color: int) -> tuple[str, np.ndarray]:
    fg = patch == int(color)
    for side, (rs, cs), marker_cells, anchor in _body_candidates(tuple(map(int, patch.shape))):
        expected = np.zeros_like(fg, dtype=bool)
        body = fg[rs, cs]
        expected[rs, cs] = body
        for rr, cc in marker_cells:
            expected[rr, cc] = True
        if not np.array_equal(fg, expected):
            continue
        if not (5 <= int(body.sum()) <= 7):
            continue
        if not body[anchor]:
            continue
        if not _connected4(body):
            continue
        return side, np.where(body, color, 0)
    raise AssertionError("component does not match the expected 3x3 body + 2-pixel directional marker structure")


def _transform_for_ring(side: str, patch: np.ndarray) -> np.ndarray:
    if side == "up":
        return np.rot90(patch, 2)
    if side == "down":
        return np.flipud(patch)
    if side == "left":
        return np.fliplr(patch)
    if side == "right":
        return np.rot90(patch, 2)
    raise ValueError(side)


def _compose_ring(patches_by_side: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((OUTPUT_SIDE, OUTPUT_SIDE), dtype=int)
    placements = {
        "up": (0, 5),
        "left": (5, 0),
        "right": (5, 8),
        "down": (8, 5),
    }
    for side, (r0, c0) in placements.items():
        _paste(out, patches_by_side[side], r0, c0, transparent=0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each marker side, place the matching sprite on that side of a ring, and reorient it so the marker points into the center.

    Input structure:
    - The grid has black background and exactly four separated monochrome sprites.
    - Each sprite is one color and consists of a 3x3 body plus a same-color
      2-pixel straight marker extending from exactly one body side.
    - The four marker sides are all different: one up, one down, one left, and one right.

    Rule:
    - Split each sprite into its 3x3 body and its outward 2-pixel marker.
    - Use the marker side to decide where that sprite belongs in the final ring:
      up-marker -> top, down-marker -> bottom, left-marker -> left, right-marker -> right.
    - Reorient the full sprite before placement so the marker faces the central
      3x3 empty square: top uses 180-degree rotation, bottom uses vertical flip,
      left uses horizontal flip, and right uses 180-degree rotation.
    - Arrange the four transformed sprites around the central 3x3 blank region;
      the corner areas remain black.

    Key insight:
    - The same visible marker feature carries two linked pieces of information:
      which ring side receives the sprite and which orientation makes that marker point inward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == 0, "expected black background"

    comps = _components(g, bg)
    assert len(comps) == 4, "expected exactly four sprites"
    assert len({int(comp["color"]) for comp in comps}) == 4, "expected four distinct sprite colors"

    patches_by_side: dict[str, np.ndarray] = {}
    for comp in comps:
        patch = _crop_bbox(g, comp["bbox"])
        side, _body = _analyze_sprite(patch, int(comp["color"]))
        assert side not in patches_by_side, "each marker side should appear exactly once"
        patches_by_side[side] = _transform_for_ring(side, patch)

    assert set(patches_by_side) == set(SIDES), "missing one or more directions"
    out = _compose_ring(patches_by_side)
    assert np.all(out[5:8, 5:8] == 0), "center must stay blank"
    return out


def _valid_bodies() -> list[np.ndarray]:
    core = np.zeros((3, 3), dtype=bool)
    core[1, 1] = True
    core[0, 1] = True
    core[2, 1] = True
    core[1, 0] = True
    core[1, 2] = True
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    bodies: list[np.ndarray] = []
    for extra_count in (1, 2):
        for extra_cells in combinations(corners, extra_count):
            body = core.copy()
            for r, c in extra_cells:
                body[r, c] = True
            if _connected4(body):
                bodies.append(body)
    assert bodies, "expected at least one usable body shape"
    return bodies


def _ensure_config_queue() -> None:
    if _CONFIG_QUEUE:
        return
    configs: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for body in _valid_bodies():
        for perm in permutations(PALETTE):
            configs.append((body.copy(), tuple(int(x) for x in perm)))
    random.shuffle(configs)
    _CONFIG_QUEUE.extend(configs)


def _build_input_patch(body: np.ndarray, side: str, color: int) -> np.ndarray:
    patch = np.zeros((5, 3), dtype=int) if side in {"up", "down"} else np.zeros((3, 5), dtype=int)
    if side == "up":
        patch[2:5, 0:3] = np.where(body, color, 0)
        patch[0, 1] = color
        patch[1, 1] = color
    elif side == "down":
        patch[0:3, 0:3] = np.where(body, color, 0)
        patch[3, 1] = color
        patch[4, 1] = color
    elif side == "left":
        patch[0:3, 2:5] = np.where(body, color, 0)
        patch[1, 0] = color
        patch[1, 1] = color
    elif side == "right":
        patch[0:3, 0:3] = np.where(body, color, 0)
        patch[1, 3] = color
        patch[1, 4] = color
    else:
        raise ValueError(side)
    return patch


def _boxes_far_enough(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int = 2) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _place_randomly(patches: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    items = list(patches.items())
    random.shuffle(items)
    for _side, patch in items:
        ph, pw = patch.shape
        candidates = [
            (r0, c0)
            for r0 in range(h - ph + 1)
            for c0 in range(w - pw + 1)
            if _boxes_far_enough((r0, c0, r0 + ph - 1, c0 + pw - 1), placed_boxes, gap=2)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _paste(grid, patch, r0, c0, transparent=0)
        placed_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    return grid


def _expected_output_from_body_and_colors(body: np.ndarray, colors_by_side: dict[str, int]) -> np.ndarray:
    transformed: dict[str, np.ndarray] = {}
    for side, color in colors_by_side.items():
        inp_patch = _build_input_patch(body, side, color)
        transformed[side] = _transform_for_ring(side, inp_patch)
    return _compose_ring(transformed)


def _output_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _validate_instance(inp: np.ndarray, expected: np.ndarray, body: np.ndarray, colors_by_side: dict[str, int]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    got = main(inp)
    if not np.array_equal(got, expected):
        return False
    if len(_components(inp, 0)) != 4:
        return False
    if int(np.count_nonzero(expected)) == 0:
        return False
    if not np.all(expected[5:8, 5:8] == 0):
        return False
    if not _transform_step_is_necessary(body, colors_by_side, expected):
        return False
    decoded_sides = set()
    for comp in _components(inp, 0):
        patch = _crop_bbox(inp, comp["bbox"])
        side, _body = _analyze_sprite(patch, int(comp["color"]))
        decoded_sides.add(side)
    if decoded_sides != set(SIDES):
        return False
    return True


def _transform_step_is_necessary(body: np.ndarray, colors_by_side: dict[str, int], expected: np.ndarray) -> bool:
    untransformed = {side: _build_input_patch(body, side, color) for side, color in colors_by_side.items()}
    naive = _compose_ring(untransformed)
    return not np.array_equal(naive, expected)


def _generate_attempt() -> np.ndarray | None:
    _ensure_config_queue()
    if not _CONFIG_QUEUE:
        return None
    body, color_perm = _CONFIG_QUEUE.pop()
    colors_by_side = dict(zip(SIDES, color_perm))
    patches = {side: _build_input_patch(body, side, color) for side, color in colors_by_side.items()}
    expected = _expected_output_from_body_and_colors(body, colors_by_side)
    if not _transform_step_is_necessary(body, colors_by_side, expected):
        return None

    for _ in range(60):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        inp = _place_randomly(patches, (h, w))
        if inp is None:
            continue
        if not _validate_instance(inp, expected, body, colors_by_side):
            continue
        sig = _output_signature(expected)
        if sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(sig)
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(INPUT_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate ring-allocation puzzle")

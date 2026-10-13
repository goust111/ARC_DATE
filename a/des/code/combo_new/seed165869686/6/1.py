from __future__ import annotations

from collections import deque

import numpy as np


SIDES = ("top", "right", "bottom", "left")
ROTATION_K = {
    "top": 0,
    "right": 3,   # clockwise 90
    "bottom": 2,
    "left": 1,    # counterclockwise 90
}

TEMPLATE_LIBRARY = [
    np.array(
        [
            [1, 0, 0],
            [1, 2, 0],
            [1, 3, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1],
            [0, 2, 0],
            [3, 3, 4],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 2, 2],
            [0, 0, 3],
            [4, 4, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [2, 2, 0, 0],
            [0, 3, 4, 0],
            [0, 3, 4, 4],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 2, 2, 0],
            [3, 0, 4, 4],
        ],
        dtype=int,
    ),
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _missing_run(side_mask: np.ndarray) -> np.ndarray | None:
    missing = np.flatnonzero(~side_mask)
    if len(missing) == 0:
        return None
    if missing[-1] - missing[0] + 1 != len(missing):
        return None
    return missing


def _parse_frame(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    if mask[1:-1, 1:-1].any():
        return None

    sides = {
        "top": mask[0, :],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }

    gap_side = None
    gap_width = None
    for side, side_mask in sides.items():
        missing = _missing_run(side_mask)
        if missing is None:
            continue
        if gap_side is not None:
            return None
        gap_side = side
        gap_width = int(len(missing))

    if gap_side is None or gap_width is None:
        return None

    for side, side_mask in sides.items():
        if side == gap_side:
            if side_mask[0] is False or side_mask[-1] is False:
                return None
        elif not side_mask.all():
            return None

    expected_area = 2 * h + 2 * w - 4 - gap_width
    if int(mask.sum()) != expected_area:
        return None

    return {
        "color": int(next(iter(colors))),
        "bbox": (r0, c0, r1, c1),
        "side": gap_side,
        "gap_width": gap_width,
    }


def _scale_pattern(pattern: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(pattern, np.ones((scale, scale), dtype=int))


def _fill_frame(out: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    ph, pw = patch.shape
    assert ph <= inner_h and pw <= inner_w
    assert (inner_h - ph) % 2 == 0 and (inner_w - pw) % 2 == 0

    out[r0 + 1 : r1, c0 + 1 : c1] = 0
    sr = r0 + 1 + (inner_h - ph) // 2
    sc = c0 + 1 + (inner_w - pw) // 2
    out[sr : sr + ph, sc : sc + pw] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular frame with a rotated and scaled copy of the loose template.

    Input structure:
    - The background is black.
    - The grid contains one small standalone multicolor template pattern.
    - It also contains several one-pixel-thick rectangular frames whose interiors are black, and each frame has one continuous gap on exactly one side.

    Rule:
    - Detect each rectangular frame and identify which side contains the gap.
    - Rotate the template by the gap side: top -> 0 degrees, right -> 90 degrees clockwise, bottom -> 180 degrees, left -> 90 degrees counterclockwise.
    - Count the gap width and use that value as the integer block-scaling factor for the rotated template.
    - Center the scaled template inside the frame interior and paint it there, leaving the rest of the grid unchanged.

    Key insight:
    - The gap encodes two visible signals at once: its side chooses the rotation, and its length chooses the scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != 0)
    frame_infos: list[dict[str, object]] = []
    template_comps: list[list[tuple[int, int]]] = []

    for comp in comps:
        info = _parse_frame(g, comp)
        if info is None:
            template_comps.append(comp)
        else:
            frame_infos.append(info)

    assert frame_infos, "expected at least one gapped frame"
    assert len(template_comps) == 1, "expected exactly one standalone template"

    tr0, tc0, tr1, tc1 = _bbox(template_comps[0])
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    out = g.copy()
    for info in frame_infos:
        side = str(info["side"])
        scale = int(info["gap_width"])
        bbox = info["bbox"]
        rotated = np.rot90(template, ROTATION_K[side])
        scaled = _scale_pattern(rotated, scale)
        _fill_frame(out, bbox, scaled)

    return out


def _draw_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    window = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != 0
    window[mask] = patch[mask]


def _recolor_template(base: np.ndarray, frame_color: int, rng: np.random.Generator) -> np.ndarray:
    placeholders = [int(v) for v in np.unique(base) if int(v) != 0]
    pool = [c for c in range(1, 10) if c != frame_color]
    rng.shuffle(pool)
    mapping = {placeholder: pool[i] for i, placeholder in enumerate(placeholders)}
    out = np.zeros_like(base)
    for placeholder, color in mapping.items():
        out[base == placeholder] = int(color)
    return out


def _frame_patch(
    inner_shape: tuple[int, int],
    frame_color: int,
    side: str,
    gap_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    inner_h, inner_w = inner_shape
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    if side in {"top", "bottom"}:
        outer_w = inner_w + 2
        start = int(rng.integers(1, outer_w - gap_width))
        row = 0 if side == "top" else inner_h + 1
        patch[row, start : start + gap_width] = 0
    else:
        outer_h = inner_h + 2
        start = int(rng.integers(1, outer_h - gap_width))
        col = 0 if side == "left" else inner_w + 1
        patch[start : start + gap_width, col] = 0

    return patch


def _candidate_positions(occupied: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_objects(
    objects: list[dict[str, object]],
    h: int,
    w: int,
    rng: np.random.Generator,
) -> list[dict[str, object]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []

    order = sorted(
        objects,
        key=lambda obj: int(np.prod(np.array(obj["patch"]).shape)),
        reverse=True,
    )
    for obj in order:
        patch = np.array(obj["patch"], dtype=int)
        positions = _candidate_positions(occupied, patch.shape)
        if not positions:
            return None
        idx = int(rng.integers(len(positions)))
        r0, c0 = positions[idx]
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = True
        placed.append({**obj, "r0": r0, "c0": c0})
    return placed


def _choose_scales(frame_count: int, rng: np.random.Generator) -> list[int]:
    if frame_count == 1:
        choices = np.array([1, 2, 3, 4], dtype=int)
    elif frame_count == 2:
        choices = np.array([1, 2, 3], dtype=int)
    else:
        choices = np.array([1, 2], dtype=int)
    return [int(rng.choice(choices)) for _ in range(frame_count)]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_color = int(rng.integers(1, 10))
    base_template = TEMPLATE_LIBRARY[int(rng.integers(len(TEMPLATE_LIBRARY)))]
    template = _recolor_template(base_template, frame_color, rng)

    frame_count = int(rng.integers(1, 5))
    directions = list(rng.permutation(np.array(SIDES, dtype=object))[:frame_count])
    scales = _choose_scales(frame_count, rng)

    frame_objects: list[dict[str, object]] = []
    for side, scale in zip(directions, scales, strict=False):
        filled = _scale_pattern(np.rot90(template, ROTATION_K[str(side)]), scale)
        frame_objects.append(
            {
                "kind": "frame",
                "patch": _frame_patch(filled.shape, frame_color, str(side), scale, rng),
                "filled": filled,
                "side": str(side),
                "scale": scale,
            }
        )

    objects: list[dict[str, object]] = frame_objects + [{"kind": "template", "patch": template}]
    max_h = max(int(np.array(obj["patch"]).shape[0]) for obj in objects)
    max_w = max(int(np.array(obj["patch"]).shape[1]) for obj in objects)
    area_need = sum((int(np.array(obj["patch"]).shape[0]) + 2) * (int(np.array(obj["patch"]).shape[1]) + 2) for obj in objects)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(area_need))))
    if min_side > 30:
        return None

    h = int(rng.integers(max(12, min_side), 31))
    w = int(rng.integers(max(12, min_side), 31))
    placed = _place_objects(objects, h, w, rng)
    if placed is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros_like(grid)
    for obj in placed:
        patch = np.array(obj["patch"], dtype=int)
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        _draw_nonzero(grid, patch, r0, c0)
        _draw_nonzero(expected, patch, r0, c0)

    for obj in placed:
        if obj["kind"] != "frame":
            continue
        filled = np.array(obj["filled"], dtype=int)
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        expected[r0 + 1 : r0 + 1 + filled.shape[0], c0 + 1 : c0 + 1 + filled.shape[1]] = filled

    if len(_components(grid != 0)) != frame_count + 1:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")

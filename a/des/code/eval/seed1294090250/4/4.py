import random
from collections import deque

import numpy as np


ORANGE = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INSTRUCTIONS = ("flip_h", "flip_v", "rot_cw", "rot_ccw")
PATTERN_CELLS = {
    "flip_h": ((0, 0), (0, 1)),
    "flip_v": ((0, 0), (1, 0)),
    "rot_cw": ((0, 0), (0, 1), (1, 1)),
    "rot_ccw": ((1, 0), (1, 1), (0, 1)),
}
PATTERN_LOOKUP = {frozenset(cells): name for name, cells in PATTERN_CELLS.items()}
ROTATE_SIZE_POOL = [
    (h, w) for h in range(5, 9) for w in range(5, 9) if (h - w) % 2 == 0
]
FLIP_SIZE_POOL = [(h, w) for h in range(5, 9) for w in range(5, 9)]
GEN_MAX_ATTEMPTS = 400
OBJECT_MAX_ATTEMPTS = 250
PLACEMENT_MAX_ATTEMPTS = 250
PALETTE = list(range(10))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    cells = np.argwhere(mask)
    r0 = int(cells[:, 0].min())
    c0 = int(cells[:, 1].min())
    r1 = int(cells[:, 0].max())
    c1 = int(cells[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(int(v) for v in cells[0])
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    mask = grid != bg
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"cells": cells, "bbox": _bbox_from_cells(cells)})
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _apply_geom(arr: np.ndarray, instruction: str) -> np.ndarray:
    if instruction == "flip_h":
        return np.fliplr(arr)
    if instruction == "flip_v":
        return np.flipud(arr)
    if instruction == "rot_cw":
        return np.rot90(arr, -1)
    if instruction == "rot_ccw":
        return np.rot90(arr, 1)
    raise AssertionError(f"unknown instruction: {instruction}")


def _aligned_origin(
    bbox: tuple[int, int, int, int], new_shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    new_h, new_w = new_shape
    top = (r0 + r1 - (new_h - 1)) // 2
    left = (c0 + c1 - (new_w - 1)) // 2
    return int(top), int(left)


def _paste_clipped(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = out.shape
    for r, c in np.argwhere(patch != bg):
        rr = top + int(r)
        cc = left + int(c)
        if 0 <= rr < h and 0 <= cc < w:
            assert out[rr, cc] == bg or out[rr, cc] == patch[r, c]
            out[rr, cc] = int(patch[r, c])


def _instruction_from_marker(marker_cells: list[tuple[int, int]]) -> str:
    rows = [r for r, _ in marker_cells]
    cols = [c for _, c in marker_cells]
    norm = frozenset((r - min(rows), c - min(cols)) for r, c in marker_cells)
    instruction = PATTERN_LOOKUP.get(norm)
    assert instruction is not None, "marker must match one of the four instruction shapes"
    assert len(norm) == len(marker_cells), "marker cells must be distinct"
    return instruction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each object contains an orange in-band instruction marker that says how to
    transform the object; the marker is erased before the transform.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert ORANGE != bg

    out = np.full_like(g, int(bg))
    objects = _components(g, bg)
    assert objects, "expected at least one object"

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = {int(v) for v in np.unique(patch) if int(v) != bg}
        assert ORANGE in colors, "each object must contain the orange marker"
        body_colors = colors - {ORANGE}
        assert len(body_colors) == 1, "each object must have one body color plus orange"
        body = next(iter(body_colors))

        marker_cells = [(int(r), int(c)) for r, c in np.argwhere(patch == ORANGE)]
        instruction = _instruction_from_marker(marker_cells)

        patch[patch == ORANGE] = int(body)
        moved = _apply_geom(patch, instruction)
        top, left = _aligned_origin((r0, c0, r1, c1), moved.shape)
        _paste_clipped(out, moved, top, left, bg)

    return out


def _interior_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    interior = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                interior.add((r, c))
    return interior


def _marker_options(mask: np.ndarray, instruction: str) -> list[list[tuple[int, int]]]:
    pattern = PATTERN_CELLS[instruction]
    max_r = max(r for r, _ in pattern)
    max_c = max(c for _, c in pattern)
    interior = _interior_cells(mask)
    options: list[list[tuple[int, int]]] = []
    for r0 in range(mask.shape[0] - max_r):
        for c0 in range(mask.shape[1] - max_c):
            cells = [(r0 + dr, c0 + dc) for dr, dc in pattern]
            if all(cell in interior for cell in cells):
                options.append(cells)
    return options


def _random_notch(mask: np.ndarray, rng: random.Random) -> None:
    h, w = mask.shape
    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        max_depth = min(2, h - 2)
        max_span = min(4, w - 1)
        if max_depth < 1 or max_span < 1:
            return
        depth = rng.randint(1, max_depth)
        span = rng.randint(1, max_span)
        start = rng.randint(0, w - span)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        mask[rows, start : start + span] = False
    else:
        max_depth = min(2, w - 2)
        max_span = min(4, h - 1)
        if max_depth < 1 or max_span < 1:
            return
        depth = rng.randint(1, max_depth)
        span = rng.randint(1, max_span)
        start = rng.randint(0, h - span)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        mask[start : start + span, cols] = False


def _build_object_spec(instruction: str, color: int, rng: random.Random) -> dict | None:
    size_pool = ROTATE_SIZE_POOL if instruction.startswith("rot") else FLIP_SIZE_POOL
    for _ in range(OBJECT_MAX_ATTEMPTS):
        h0, w0 = rng.choice(size_pool)
        mask = np.ones((h0, w0), dtype=bool)
        for _ in range(rng.randint(2, 5)):
            _random_notch(mask, rng)

        corner_r = 0 if rng.random() < 0.5 else h0 - 1
        corner_c = 0 if rng.random() < 0.5 else w0 - 1
        mask[corner_r, corner_c] = False

        if not bool(mask.any()):
            continue
        mask = _crop_mask(mask)
        h, w = mask.shape
        if not (5 <= h <= 8 and 5 <= w <= 8):
            continue
        if instruction.startswith("rot") and (h - w) % 2 != 0:
            continue
        if int(mask.sum()) < 18:
            continue
        density = float(mask.mean())
        if density < 0.5 or density > 0.88:
            continue
        if not _is_connected(mask):
            continue
        marker_options = _marker_options(mask, instruction)
        if not marker_options:
            continue
        if np.array_equal(mask, _apply_geom(mask, instruction)):
            continue

        marker = rng.choice(marker_options)
        patch = np.full((h, w), -1, dtype=int)
        patch[mask] = int(color)
        for r, c in marker:
            patch[r, c] = ORANGE
        patch[patch == -1] = -99
        return {
            "instruction": instruction,
            "color": int(color),
            "mask": mask,
            "marker": tuple(marker),
            "local_patch": patch,
        }
    return None


def _materialize_patch(spec: dict, bg: int) -> np.ndarray:
    patch = spec["local_patch"].copy()
    patch[patch == -99] = int(bg)
    return patch


def _bbox_in_bounds(bbox: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _out_bbox(in_bbox: tuple[int, int, int, int], instruction: str) -> tuple[int, int, int, int]:
    h = in_bbox[2] - in_bbox[0] + 1
    w = in_bbox[3] - in_bbox[1] + 1
    moved_shape = _apply_geom(np.zeros((h, w), dtype=int), instruction).shape
    top, left = _aligned_origin(in_bbox, moved_shape)
    return (top, left, top + moved_shape[0] - 1, left + moved_shape[1] - 1)


def _choose_grid_shape(object_count: int, rng: random.Random) -> tuple[int, int]:
    if object_count == 4:
        return rng.randint(22, 27), rng.randint(22, 28)
    return rng.randint(24, 30), rng.randint(24, 30)


def _build_specs(object_count: int, bg: int, rng: random.Random) -> list[dict] | None:
    colors = [c for c in PALETTE if c not in (bg, ORANGE)]
    rng.shuffle(colors)
    chosen_colors = colors[:object_count]
    instructions = list(INSTRUCTIONS)
    if object_count > 4:
        instructions.append(rng.choice(INSTRUCTIONS))
    rng.shuffle(instructions)

    specs: list[dict] = []
    for instruction, color in zip(instructions, chosen_colors):
        spec = _build_object_spec(instruction, color, rng)
        if spec is None:
            return None
        specs.append(spec)
    return specs


def _scatter_specs(specs: list[dict], bg: int, rng: random.Random) -> np.ndarray | None:
    h, w = _choose_grid_shape(len(specs), rng)
    grid = np.full((h, w), int(bg), dtype=int)
    placed: list[dict] = []

    order = sorted(
        range(len(specs)),
        key=lambda idx: (
            -max(specs[idx]["local_patch"].shape),
            -min(specs[idx]["local_patch"].shape),
        ),
    )

    for idx in order:
        spec = specs[idx]
        patch = _materialize_patch(spec, bg)
        ph, pw = patch.shape
        for _ in range(PLACEMENT_MAX_ATTEMPTS):
            top = rng.randint(0, h - ph)
            left = rng.randint(0, w - pw)
            in_bbox = (top, left, top + ph - 1, left + pw - 1)
            out_bbox = _out_bbox(in_bbox, spec["instruction"])
            if not _bbox_in_bounds(out_bbox, h, w):
                continue
            if any(not _boxes_separated(in_bbox, item["in_bbox"], 1) for item in placed):
                continue
            if any(not _boxes_separated(out_bbox, item["out_bbox"], 1) for item in placed):
                continue
            target = grid[top : top + ph, left : left + pw]
            mask = patch != bg
            if np.any(target[mask] != bg):
                continue
            target[mask] = patch[mask]
            placed.append({"in_bbox": in_bbox, "out_bbox": out_bbox, "idx": idx})
            spec["in_bbox"] = in_bbox
            break
        else:
            return None

    return grid


def _valid_generated_case(grid: np.ndarray, specs: list[dict]) -> bool:
    if int((grid == ORANGE).sum()) != sum(len(spec["marker"]) for spec in specs):
        return False
    bg = get_bg(grid)
    comps = _components(grid, bg)
    if len(comps) != len(specs):
        return False
    instructions_found = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(patch) if int(v) != bg}
        if ORANGE not in colors or len(colors - {ORANGE}) != 1:
            return False
        marker_cells = [(int(r), int(c)) for r, c in np.argwhere(patch == ORANGE)]
        instructions_found.append(_instruction_from_marker(marker_cells))
    if not all(name in instructions_found for name in INSTRUCTIONS):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if len(_components(out, bg)) != len(specs):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rng = random.Random()
        object_count = 4 if rng.random() < 0.65 else 5
        bg_choices = [c for c in PALETTE if c != ORANGE]
        bg = rng.choice(bg_choices)
        specs = _build_specs(object_count, bg, rng)
        if specs is None:
            continue
        grid = _scatter_specs(specs, bg, rng)
        if grid is None:
            continue
        if _valid_generated_case(grid, specs):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")

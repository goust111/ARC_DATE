from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

OUTPUT_COLORS = (RED, BLUE, GREEN, YELLOW)


_RNG = np.random.default_rng()

# --------------- shared helpers ---------------


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "mask must be non-empty"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0) + 1
    return np.asarray(mask[r0:r1, c0:c1], dtype=bool)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    arr = np.asarray(coords, dtype=int)
    (r0, c0), (r1, c1) = arr.min(axis=0), arr.max(axis=0) + 1
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = _trim_mask(mask)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c, h, w):
                if not mask[nr, nc]:
                    out[r, c] = True
                    break
            else:
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    out[r, c] = True
    return out


def _has_interior_cell(mask: np.ndarray) -> bool:
    mask = _trim_mask(mask)
    h, w = mask.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                return True
    return False


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return len(_component_coords(mask)) == 1


def _dihedral_transforms(mask: np.ndarray) -> list[np.ndarray]:
    base = _trim_mask(mask)
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(base, k)
        out.append(_trim_mask(rot))
        out.append(_trim_mask(np.fliplr(rot)))
    return out


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    trimmed = _trim_mask(mask)
    flat = tuple(int(v) for v in trimmed.astype(np.uint8).ravel())
    return trimmed.shape[0], trimmed.shape[1], flat


def _canonical_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return min(_mask_key(t) for t in _dihedral_transforms(mask))


def _recover_filled_from_outline(outline: np.ndarray) -> np.ndarray:
    outline = _trim_mask(outline)
    h, w = outline.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue = deque()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if outline[r, c] or exterior[r, c]:
                continue
            exterior[r, c] = True
            queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if outline[nr, nc] or exterior[nr, nc]:
                continue
            exterior[nr, nc] = True
            queue.append((nr, nc))

    interior = (~outline) & (~exterior)
    return _trim_mask(outline | interior)


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey separator"
    return cols[0]


def _extract_objects(panel: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(panel)):
        if color in (BLACK, GREY, MAROON):
            continue
        for coords in _component_coords(panel == color):
            mask, bbox = _coords_to_mask(coords)
            objects.append({"color": color, "bbox": bbox, "mask": mask})
    return objects


# --------------- main ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the left panel unchanged, then copy only the solid objects whose shape
    matches the maroon hollow template up to rotation/reflection into the right
    panel. Recolor the copied shapes with a fixed cyclic palette and stack them
    top-to-bottom in the template's orientation.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape[1] == right.shape[1], "panels must have equal width"

    template_outline = left == MAROON
    assert np.any(template_outline), "left panel must contain one maroon template"
    template_mask = _recover_filled_from_outline(template_outline)
    template_sig = _canonical_signature(template_mask)
    th, tw = template_mask.shape

    matches: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for obj in _extract_objects(left):
        mask = np.asarray(obj["mask"], dtype=bool)
        if _canonical_signature(mask) == template_sig:
            matches.append((obj["bbox"], mask))

    matches.sort(key=lambda item: (item[0][0], item[0][1]))

    out = g.copy()
    out[:, sep + 1 :] = BLACK
    cursor = 0
    for idx, (_bbox, _mask) in enumerate(matches):
        assert cursor + th <= out.shape[0], "stacked output must fit the right panel"
        assert tw <= right.shape[1], "template width must fit the right panel"
        patch = out[cursor : cursor + th, sep + 1 : sep + 1 + tw]
        patch[template_mask] = OUTPUT_COLORS[idx % len(OUTPUT_COLORS)]
        cursor += th + 1

    return out


# --------------- generator ---------------


GEN_ATTEMPTS = 500
GEN_PANEL_H_MAX = 30
GEN_PANEL_W_MIN = 13
GEN_PANEL_W_MAX = 19
GEN_MATCH_COUNTS = np.array([1, 2, 3, 4, 5], dtype=int)
GEN_MATCH_WEIGHTS = np.array([0.14, 0.24, 0.24, 0.22, 0.16], dtype=float)
GEN_OBJECT_COLORS = np.array([BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN], dtype=int)


def _unique_transform_count(mask: np.ndarray) -> int:
    return len({_mask_key(t) for t in _dihedral_transforms(mask)})


def _random_filled_shape(
    rng: np.random.Generator,
    min_side: int,
    max_side: int,
    min_area: int,
    require_interior: bool,
) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(min_side, max_side + 1))
        w = int(rng.integers(min_side, max_side + 1))
        mask = np.ones((h, w), dtype=bool)
        steps = int(rng.integers(max(h, w), h * w + 1))
        carved = 0

        for _ in range(steps):
            boundary = np.argwhere(_boundary_mask(mask))
            if len(boundary) <= 4:
                break
            rr, cc = boundary[int(rng.integers(len(boundary)))]
            trial = mask.copy()
            trial[rr, cc] = False
            trial = _trim_mask(trial)
            if int(trial.sum()) < min_area:
                continue
            if not _is_connected(trial):
                continue
            if require_interior and not _has_interior_cell(trial):
                continue
            mask = trial
            carved += 1

        mask = _trim_mask(mask)
        if int(mask.sum()) < min_area:
            continue
        if require_interior and not _has_interior_cell(mask):
            continue
        if require_interior and _unique_transform_count(mask) < 2:
            continue
        if not require_interior and carved == 0 and rng.random() < 0.75:
            continue
        return mask
    raise RuntimeError("failed to build a random shape")


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(0, 4))
    out = np.rot90(mask, k)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return _trim_mask(out)


def _place_item(
    canvas: np.ndarray,
    reserve: np.ndarray,
    paint_mask: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = paint_mask.shape
    H, W = canvas.shape
    for _ in range(200):
        top = int(rng.integers(0, H - h + 1))
        left = int(rng.integers(0, W - w + 1))
        r0 = max(0, top - 1)
        r1 = min(H, top + h + 1)
        c0 = max(0, left - 1)
        c1 = min(W, left + w + 1)
        if reserve[r0:r1, c0:c1].any():
            continue
        canvas[top : top + h, left : left + w][paint_mask] = color
        reserve[r0:r1, c0:c1] = True
        return top, left
    return None


def _gen_attempt() -> np.ndarray | None:
    template = _random_filled_shape(_RNG, min_side=3, max_side=5, min_area=6, require_interior=True)
    template_sig = _canonical_signature(template)
    outline = _boundary_mask(template)
    th, tw = template.shape

    match_count = int(_RNG.choice(GEN_MATCH_COUNTS, p=GEN_MATCH_WEIGHTS))
    max_distractors = min(4, len(GEN_OBJECT_COLORS) - match_count)
    if max_distractors < 1:
        return None
    distractor_count = int(_RNG.integers(1, max_distractors + 1))
    stack_height = match_count * th + (match_count - 1)
    if stack_height > GEN_PANEL_H_MAX:
        return None

    panel_h_min = max(12, stack_height)
    panel_h = int(_RNG.integers(panel_h_min, GEN_PANEL_H_MAX + 1))
    panel_w_min = max(GEN_PANEL_W_MIN, tw + 4)
    panel_w = int(_RNG.integers(panel_w_min, GEN_PANEL_W_MAX + 1))

    total_objects = match_count + distractor_count
    colors = [int(c) for c in _RNG.choice(GEN_OBJECT_COLORS, size=total_objects, replace=False)]

    items: list[dict[str, object]] = [
        {"paint_mask": outline, "bbox_shape": template.shape, "color": MAROON, "is_match": False}
    ]

    for idx in range(match_count):
        items.append(
            {
                "paint_mask": _random_transform(template, _RNG),
                "bbox_shape": None,
                "color": colors[idx],
                "is_match": True,
            }
        )

    distractors: list[np.ndarray] = []
    for _ in range(distractor_count):
        for _ in range(200):
            if _RNG.random() < 0.5:
                candidate = _random_filled_shape(_RNG, min_side=3, max_side=5, min_area=5, require_interior=True)
            else:
                candidate = _random_filled_shape(_RNG, min_side=2, max_side=6, min_area=4, require_interior=False)
            if _canonical_signature(candidate) != template_sig:
                distractors.append(candidate)
                break
        else:
            return None

    for idx, distractor in enumerate(distractors, start=match_count):
        items.append(
            {
                "paint_mask": _random_transform(distractor, _RNG),
                "bbox_shape": None,
                "color": colors[idx],
                "is_match": False,
            }
        )

    left = np.full((panel_h, panel_w), BLACK, dtype=int)
    reserve = np.zeros((panel_h, panel_w), dtype=bool)
    order = list(range(len(items)))
    _RNG.shuffle(order)
    order.sort(
        key=lambda idx: (
            items[idx]["paint_mask"].shape[0] * items[idx]["paint_mask"].shape[1],
            int(items[idx]["paint_mask"].sum()),
        ),
        reverse=True,
    )

    for idx in order:
        item = items[idx]
        pos = _place_item(
            left,
            reserve,
            np.asarray(item["paint_mask"], dtype=bool),
            int(item["color"]),
            _RNG,
        )
        if pos is None:
            return None
        item["pos"] = pos

    grid = np.full((panel_h, panel_w * 2 + 1), BLACK, dtype=int)
    grid[:, :panel_w] = left
    grid[:, panel_w] = GREY

    # Independent validation of the intended input structure.
    recovered = _recover_filled_from_outline(left == MAROON)
    if not np.array_equal(recovered, template):
        return None

    actual_matches = 0
    for obj in _extract_objects(left):
        if _canonical_signature(np.asarray(obj["mask"], dtype=bool)) == template_sig:
            actual_matches += 1
    if actual_matches != match_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

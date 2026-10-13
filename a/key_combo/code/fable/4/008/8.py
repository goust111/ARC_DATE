from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8

PROTOTYPE_COLORS = (1, 2, 3, 4, 6, 7, 9)
OBJECT_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 800


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_by_color(
    grid: np.ndarray,
    *,
    only_color: int | None = None,
) -> list[dict[str, object]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict[str, object]] = []

    for sr in range(height):
        for sc in range(width):
            color = int(grid[sr, sc])
            if seen[sr, sc] or color == BLACK:
                continue
            if only_color is not None and color != int(only_color):
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))

            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_corner_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    height, width = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == width - 1)
        or (r1 == height - 1 and c0 == 0)
        or (r1 == height - 1 and c1 == width - 1)
    )


def _is_legend_frame(grid: np.ndarray, comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False
    if not _is_corner_bbox((int(r0), int(c0), int(r1), int(c1)), grid.shape):
        return False

    coords = comp["coords"]  # type: ignore[assignment]
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    expected = 2 * height + 2 * width - 4
    if len(rel) != expected:
        return False

    for rr in range(height):
        for cc in range(width):
            on_frame = rr == 0 or rr == height - 1 or cc == 0 or cc == width - 1
            if on_frame != ((rr, cc) in rel):
                return False

    interior = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    fg = interior[interior != BLACK]
    return bool(fg.size) and len(set(map(int, fg))) == 1


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frames = [
        comp["bbox"]
        for comp in _components_by_color(grid, only_color=GREY)
        if _is_legend_frame(grid, comp)
    ]
    assert len(frames) == 1, "expected a unique grey legend frame in one corner"
    return tuple(map(int, frames[0]))  # type: ignore[arg-type]


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _compress_mask(mask: np.ndarray) -> np.ndarray:
    """Fold adjacent equal row and column runs until both axes are stable."""
    out = np.asarray(mask, dtype=bool).copy()
    assert out.ndim == 2 and bool(out.any()), "mask must be non-empty"

    while True:
        changed = False
        if out.shape[0] > 1:
            keep_rows = [0]
            for r in range(1, out.shape[0]):
                if not np.array_equal(out[r], out[r - 1]):
                    keep_rows.append(r)
            if len(keep_rows) != out.shape[0]:
                out = out[keep_rows, :]
                changed = True

        if out.shape[1] > 1:
            keep_cols = [0]
            for c in range(1, out.shape[1]):
                if not np.array_equal(out[:, c], out[:, c - 1]):
                    keep_cols.append(c)
            if len(keep_cols) != out.shape[1]:
                out = out[:, keep_cols]
                changed = True

        if not changed:
            return out


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            cand = np.asarray(candidate, dtype=bool)
            if not any(cand.shape == old.shape and np.array_equal(cand, old) for old in variants):
                variants.append(cand.copy())
    return variants


def _matches_any(mask: np.ndarray, variants: list[np.ndarray]) -> bool:
    return any(mask.shape == v.shape and np.array_equal(mask, v) for v in variants)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count field objects topologically equivalent to the corner legend prototype.

    Input structure:
    - The background is black.
    - One corner contains a grey rectangular legend frame whose interior holds a
      single-color prototype symbol.
    - Outside the legend are separated monochromatic candidate objects and
      distractors, with colors that do not determine the answer.

    Rule:
    - Remove the legend from the field and examine every monochromatic
      4-connected object.
    - Crop each object to its bounding box, fold adjacent repeated rows and
      columns in that mask, and do the same to the legend prototype.
    - Count the objects whose compressed mask equals any rotation or reflection
      of the compressed prototype mask, ignoring color.
    - Return a 1 by k row of teal cells, where k is that count.

    Key insight:
    - Color and raw pixel scale are decoys; the invariant is the compressed
      topology of the shape up to the eight rotation/reflection symmetries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _legend_bbox(g)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    proto_mask = interior != BLACK
    assert bool(proto_mask.any()), "legend prototype must be non-empty"
    assert len(set(map(int, interior[proto_mask]))) == 1, "prototype must be single-color"

    proto = _compress_mask(proto_mask)
    proto_variants = _orientations(proto)

    field = g.copy()
    field[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    count = 0
    for comp in _components_by_color(field):
        mask = _component_mask(
            comp["coords"],  # type: ignore[arg-type]
            comp["bbox"],  # type: ignore[arg-type]
        )
        if _matches_any(_compress_mask(mask), proto_variants):
            count += 1

    assert count <= 30, "answer row must fit ARC dimensions"
    return np.full((1, count), TEAL, dtype=int)


def _bool_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_MASKS = (
    _bool_grid(("110", "010", "011")),
    _bool_grid(("1110", "0100", "0111")),
    _bool_grid(("100", "111", "001")),
    _bool_grid(("011", "010", "110", "100")),
    _bool_grid(("010", "111", "100")),
)


def _transform(mask: np.ndarray, transform_id: int) -> np.ndarray:
    out = np.rot90(mask, transform_id % 4)
    if transform_id >= 4:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool).copy()


def _stretch_mask(mask: np.ndarray, *, force: bool = False) -> tuple[np.ndarray, bool]:
    row_runs = [2 if random.random() < 0.35 else 1 for _ in range(mask.shape[0])]
    col_runs = [2 if random.random() < 0.35 else 1 for _ in range(mask.shape[1])]

    if force and max(row_runs + col_runs) == 1:
        if random.choice((True, False)):
            row_runs[random.randrange(len(row_runs))] = 2
        else:
            col_runs[random.randrange(len(col_runs))] = 2

    stretched = max(row_runs + col_runs) > 1
    out = np.repeat(np.repeat(mask, row_runs, axis=0), col_runs, axis=1)
    return np.asarray(out, dtype=bool), bool(stretched)


def _has_adjacent_duplicate(mask: np.ndarray) -> bool:
    for r in range(1, mask.shape[0]):
        if np.array_equal(mask[r], mask[r - 1]):
            return True
    for c in range(1, mask.shape[1]):
        if np.array_equal(mask[:, c], mask[:, c - 1]):
            return True
    return False


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    height, width = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((int(nr), int(nc)))
                q.append((int(nr), int(nc)))
    return len(seen) == int(mask.sum())


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        area = random.randint(2, min(8, height * width))
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        b = _bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if _connected(mask):
            return mask
    raise RuntimeError("failed to make connected distractor")


def _distractor_mask(proto_variants: list[np.ndarray]) -> np.ndarray:
    for _ in range(300):
        mask = _random_connected_mask()
        compressed = _compress_mask(mask)
        if compressed.sum() < 2:
            continue
        if not _matches_any(compressed, proto_variants):
            return mask
    raise RuntimeError("failed to make non-equivalent distractor")


def _legend_patch(proto_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((proto_mask.shape[0] + 2, proto_mask.shape[1] + 2), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1:-1, 1:-1][proto_mask] = int(color)
    return patch


def _color_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    view = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _corner_position(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    corner: str,
) -> tuple[int, int]:
    height, width = shape
    ph, pw = patch_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - pw
    if corner == "bl":
        return height - ph, 0
    return height - ph, width - pw


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    height, width = canvas.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > height or c0 + pw > width:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(height, r0 + ph + padding)
    cc1 = min(width, c0 + pw + padding)
    return bool(np.all(canvas[rr0:rr1, cc0:cc1] == BLACK))


def _place_object(canvas: np.ndarray, patch: np.ndarray) -> bool:
    height, width = canvas.shape
    ph, pw = patch.shape
    candidates = [
        (r, c)
        for r in range(height - ph + 1)
        for c in range(width - pw + 1)
        if _can_place(canvas, patch, r, c)
    ]
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    _paste(canvas, patch, r0, c0)
    return True


def _match_object(
    proto: np.ndarray,
    *,
    require_nonidentity: bool = False,
    require_stretch: bool = False,
) -> tuple[np.ndarray, bool, bool]:
    transform_ids = list(range(8))
    if require_nonidentity:
        transform_ids = [
            i
            for i in transform_ids
            if not (_transform(proto, i).shape == proto.shape and np.array_equal(_transform(proto, i), proto))
        ]
    transform_id = random.choice(transform_ids or list(range(8)))
    transformed = _transform(proto, transform_id)
    mask, stretched = _stretch_mask(transformed, force=require_stretch)
    nonidentity = not (transformed.shape == proto.shape and np.array_equal(transformed, proto))
    return mask, bool(nonidentity), bool(stretched or _has_adjacent_duplicate(mask))


def _build_records(proto: np.ndarray, proto_color: int) -> list[dict[str, object]]:
    proto_variants = _orientations(proto)
    count = random.randint(2, 5)
    match_colors = [c for c in OBJECT_COLORS if c != int(proto_color)]

    records: list[dict[str, object]] = []
    mask, nonidentity, stretched = _match_object(
        proto,
        require_nonidentity=True,
        require_stretch=True,
    )
    records.append(
        {
            "mask": mask,
            "color": random.choice(match_colors),
            "match": True,
            "nonidentity": nonidentity,
            "stretched": stretched,
        }
    )

    for _ in range(count - 1):
        mask, nonidentity, stretched = _match_object(
            proto,
            require_nonidentity=False,
            require_stretch=random.random() < 0.45,
        )
        records.append(
            {
                "mask": mask,
                "color": random.choice(match_colors),
                "match": True,
                "nonidentity": nonidentity,
                "stretched": stretched,
            }
        )

    same_color_decoy = _distractor_mask(proto_variants)
    records.append(
        {
            "mask": same_color_decoy,
            "color": int(proto_color),
            "match": False,
            "nonidentity": False,
            "stretched": _has_adjacent_duplicate(same_color_decoy),
        }
    )

    for _ in range(random.randint(1, 3)):
        distractor = _distractor_mask(proto_variants)
        color = random.choice(match_colors)
        records.append(
            {
                "mask": distractor,
                "color": int(color),
                "match": False,
                "nonidentity": False,
                "stretched": _has_adjacent_duplicate(distractor),
            }
        )

    for record in records:
        compressed = _compress_mask(record["mask"])  # type: ignore[arg-type]
        assert _matches_any(compressed, proto_variants) == bool(record["match"])

    random.shuffle(records)
    return records


def _semantic_checks(records: list[dict[str, object]], proto_color: int) -> bool:
    answer = sum(1 for record in records if bool(record["match"]))
    total = len(records)
    proto_color_objects = sum(1 for record in records if int(record["color"]) == int(proto_color))
    if answer == total or answer == proto_color_objects:
        return False
    if not any(bool(r["match"]) and bool(r["stretched"]) for r in records):
        return False
    if not any(bool(r["match"]) and bool(r["nonidentity"]) for r in records):
        return False
    if not any(bool(r["match"]) and int(r["color"]) != int(proto_color) for r in records):
        return False
    if not any((not bool(r["match"])) and int(r["color"]) == int(proto_color) for r in records):
        return False
    if not any((not bool(r["match"])) and int(r["color"]) != int(proto_color) for r in records):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    core = random.choice(BASE_MASKS)
    proto_display, _ = _stretch_mask(core, force=random.random() < 0.35)
    proto = _compress_mask(proto_display)
    proto_color = random.choice(PROTOTYPE_COLORS)
    legend = _legend_patch(proto_display, proto_color)

    min_rows = max(22, legend.shape[0] + 12)
    min_cols = max(22, legend.shape[1] + 12)
    rows = random.randint(min_rows, 30)
    cols = random.randint(min_cols, 30)
    grid = np.zeros((rows, cols), dtype=int)

    corner = random.choice(("tl", "tr", "bl", "br"))
    lr, lc = _corner_position(grid.shape, legend.shape, corner)
    _paste(grid, legend, lr, lc)

    records = _build_records(proto, proto_color)
    if not _semantic_checks(records, proto_color):
        return None

    patches = [
        _color_patch(record["mask"], int(record["color"]))  # type: ignore[arg-type]
        for record in records
    ]
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        if not _place_object(grid, patches[idx]):
            return None

    expected_count = sum(1 for record in records if bool(record["match"]))
    expected = np.full((1, expected_count), TEAL, dtype=int)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topology-equivalence counting input")

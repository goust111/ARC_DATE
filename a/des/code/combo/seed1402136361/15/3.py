from __future__ import annotations

from collections import Counter
import random

import numpy as np


EDGE_PRIORITY = ("top", "bottom", "left", "right")
EDGE_INDEX = {edge: idx for idx, edge in enumerate(EDGE_PRIORITY)}
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
GRID_SIZE_RANGE = (16, 24)
OBJECT_COUNT_OPTIONS = (2, 3, 4, 5)
OBJECT_HEIGHT_RANGE = (2, 5)
OBJECT_WIDTH_RANGE = (2, 5)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": _bbox(coords),
                }
            )

    return comps


def _nearest_edge(
    bbox: tuple[int, int, int, int], height: int, width: int
) -> tuple[str, dict[str, int]]:
    r0, c0, r1, c1 = bbox
    distances = {
        "top": int(r0),
        "bottom": int(height - 1 - r1),
        "left": int(c0),
        "right": int(width - 1 - c1),
    }
    edge = min(EDGE_PRIORITY, key=lambda name: (distances[name], EDGE_INDEX[name]))
    return edge, distances


def main(grid: np.ndarray) -> np.ndarray:
    """Project each foreground object to its nearest grid edge."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = get_bg(g)
    out = np.full_like(g, bg)

    for comp in _components(g, bg):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        edge, _distances = _nearest_edge(comp["bbox"], height, width)

        if edge == "top":
            out[0, c0 : c1 + 1] = color
        elif edge == "bottom":
            out[height - 1, c0 : c1 + 1] = color
        elif edge == "left":
            out[r0 : r1 + 1, 0] = color
        else:
            out[r0 : r1 + 1, width - 1] = color

    return out


def _make_shape(height: int, width: int) -> np.ndarray:
    mode = random.choice(("frame", "ell", "tee", "plus", "u", "h", "rect"))
    mask = np.zeros((height, width), dtype=bool)

    if mode == "frame":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    elif mode == "ell":
        mask[random.choice((0, height - 1)), :] = True
        mask[:, random.choice((0, width - 1))] = True
    elif mode == "tee":
        mask[random.choice((0, height - 1)), :] = True
        mask[:, width // 2] = True
    elif mode == "plus":
        mask[height // 2, :] = True
        mask[:, width // 2] = True
    elif mode == "u":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[random.choice((0, height - 1)), :] = True
    elif mode == "h":
        mask[:, 0] = True
        mask[:, -1] = True
        mask[height // 2, :] = True
    else:
        mask[:, :] = True

    return mask


def _bboxes_too_close(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _edge_span(bbox: tuple[int, int, int, int], edge: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if edge in {"top", "bottom"}:
        return c0, c1
    return r0, r1


def _sample_bbox(
    height: int, width: int, obj_h: int, obj_w: int, edge: str
) -> tuple[int, int, int, int] | None:
    if edge in {"top", "bottom"}:
        max_d = min(3, (height - obj_h - 1) // 2, (width - obj_w - 2) // 2)
        if max_d < 1:
            return None
        d = random.randint(1, max_d)
        c_min = d + 1
        c_max = width - obj_w - d - 1
        if c_min > c_max:
            return None
        c0 = random.randint(c_min, c_max)
        r0 = d if edge == "top" else height - obj_h - d
    else:
        max_d = min(3, (width - obj_w - 1) // 2, (height - obj_h - 2) // 2)
        if max_d < 1:
            return None
        d = random.randint(1, max_d)
        r_min = d + 1
        r_max = height - obj_h - d - 1
        if r_min > r_max:
            return None
        r0 = random.randint(r_min, r_max)
        c0 = d if edge == "left" else width - obj_w - d
    return r0, c0, r0 + obj_h - 1, c0 + obj_w - 1


def _sample_edges(count: int) -> list[str]:
    edges = random.sample(list(EDGE_PRIORITY), k=min(4, count))
    while len(edges) < count:
        edges.append(random.choice(EDGE_PRIORITY))

    if count >= 3 and random.random() < 0.45:
        duplicate = random.choice(EDGE_PRIORITY)
        replace_at = random.randrange(count)
        edges[replace_at] = duplicate
        if len(set(edges)) == 1:
            edges[(replace_at + 1) % count] = random.choice(
                [edge for edge in EDGE_PRIORITY if edge != duplicate]
            )

    random.shuffle(edges)
    return edges


def _foreground_area(masked_objects: list[dict]) -> int:
    return sum(int(obj["mask"].sum()) for obj in masked_objects)


def _validate_grid(grid: np.ndarray) -> bool:
    bg = get_bg(grid)
    comps = _components(grid, bg)
    if not comps:
        return False
    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return False

    by_color = Counter(int(comp["color"]) for comp in comps)
    if any(count != 1 for count in by_color.values()):
        return False

    spans_by_edge = {edge: [] for edge in EDGE_PRIORITY}
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1:
            return False

        edge, distances = _nearest_edge(comp["bbox"], *grid.shape)
        best = distances[edge]
        if sum(1 for distance in distances.values() if distance == best) != 1:
            return False

        span = _edge_span(comp["bbox"], edge)
        if any(_intervals_overlap(span, other) for other in spans_by_edge[edge]):
            return False
        spans_by_edge[edge].append(span)

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    bg = random.randrange(10)
    obj_count = random.choice(OBJECT_COUNT_OPTIONS)
    edges = _sample_edges(obj_count)
    colors = random.sample([color for color in range(10) if color != bg], k=obj_count)

    objects: list[dict] = []
    spans_by_edge = {edge: [] for edge in EDGE_PRIORITY}

    for edge, color in zip(edges, colors):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            obj_h = random.randint(OBJECT_HEIGHT_RANGE[0], min(OBJECT_HEIGHT_RANGE[1], height - 4))
            obj_w = random.randint(OBJECT_WIDTH_RANGE[0], min(OBJECT_WIDTH_RANGE[1], width - 4))
            bbox = _sample_bbox(height, width, obj_h, obj_w, edge)
            if bbox is None:
                continue
            if any(_bboxes_too_close(bbox, other["bbox"]) for other in objects):
                continue

            span = _edge_span(bbox, edge)
            if any(_intervals_overlap(span, other) for other in spans_by_edge[edge]):
                continue

            mask = _make_shape(obj_h, obj_w)
            objects.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "mask": mask,
                }
            )
            spans_by_edge[edge].append(span)
            placed = True
            break

        if not placed:
            return None

    if _foreground_area(objects) * 2 >= height * width:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        mask = obj["mask"]
        grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(obj["color"])

    if not _validate_grid(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

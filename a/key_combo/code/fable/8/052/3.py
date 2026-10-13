from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


CoordArray = np.ndarray


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if int(connectivity) == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )


def _components_mask(mask: np.ndarray, connectivity: int) -> list[CoordArray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[CoordArray] = []
    steps = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: CoordArray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shift(coords: CoordArray, dr: int, dc: int) -> CoordArray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _coords_to_mask(shape: tuple[int, int], coords: CoordArray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _touches_edge(coords: CoordArray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _holes_in_component(mask: np.ndarray, coords: CoordArray) -> list[CoordArray]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[CoordArray] = []
    for hole in _components_mask(~sub, connectivity=4):
        if _touches_edge(hole, sub.shape):
            continue
        shifted = hole.copy()
        shifted[:, 0] += int(r0)
        shifted[:, 1] += int(c0)
        holes.append(shifted)
    return holes


def _component_sort_key(coords: CoordArray, direction: int) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == -1:
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    return (-int(cols.max()), int(rows.min()), -int(cols.min()), int(rows.max()))


def _parse_objects(grid: np.ndarray) -> tuple[int, int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])

    assert left_color != right_color, "left and right walls must have different colors"
    assert bg not in {left_color, right_color}, "walls must be non-background colors"
    assert np.all(g[:, 0] == left_color), "left wall must fill column 0"
    assert np.all(g[:, w - 1] == right_color), "right wall must fill last column"

    colors = {int(v) for v in np.unique(g)}
    inner_colors = sorted(colors - {bg, left_color, right_color})
    assert len(inner_colors) == 1, "expected exactly one hollow-object color"
    object_color = int(inner_colors[0])

    object_mask = (g == object_color)
    components = _components_mask(object_mask, connectivity=8)
    assert len(components) >= 3, "expected multiple hollow objects"

    objects: list[dict] = []
    for coords in components:
        holes = _holes_in_component(object_mask, coords)
        assert holes, "each object must enclose at least one hole"
        hole_area = sum(int(hole.shape[0]) for hole in holes)
        target_side = "left" if hole_area % 2 == 0 else "right"
        bbox = _bbox(coords)
        body_rel = coords - np.array([bbox[0], bbox[1]], dtype=int)
        hole_rels = [hole - np.array([bbox[0], bbox[1]], dtype=int) for hole in holes]
        silhouette_rel = np.vstack([body_rel] + hole_rels)
        objects.append(
            {
                "coords": coords,
                "holes": holes,
                "hole_area": int(hole_area),
                "target_side": target_side,
                "bbox": bbox,
                "body_rel": body_rel,
                "hole_rels": hole_rels,
                "silhouette_rel": silhouette_rel,
                "shape": (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1),
                "distance": int(coords[:, 1].min()) if target_side == "left" else int((w - 1) - coords[:, 1].max()),
            }
        )
    return bg, left_color, right_color, object_color, objects


def _settle_side(
    shape: tuple[int, int],
    objects: Iterable[dict],
    side_col: int,
    direction: int,
) -> list[dict]:
    h, w = shape
    settled = np.zeros((h, w), dtype=bool)
    settled[:, int(side_col)] = True
    placed: list[dict] = []

    for obj in sorted(objects, key=lambda item: _component_sort_key(item["coords"], direction)):
        bbox = obj["bbox"]
        r0, c0 = bbox[0], bbox[1]
        shift = 0
        while True:
            candidate_sil = _shift(obj["silhouette_rel"], r0, c0 + shift + direction)
            if int(candidate_sil[:, 1].min()) < 0 or int(candidate_sil[:, 1].max()) >= w:
                break
            if np.any(settled[candidate_sil[:, 0], candidate_sil[:, 1]]):
                break
            shift += direction

        out_body = _shift(obj["body_rel"], r0, c0 + shift)
        out_holes = [_shift(hole_rel, r0, c0 + shift) for hole_rel in obj["hole_rels"]]
        out_sil = _shift(obj["silhouette_rel"], r0, c0 + shift)
        assert not np.any(settled[out_sil[:, 0], out_sil[:, 1]]), "settled silhouettes must not overlap"
        settled[out_sil[:, 0], out_sil[:, 1]] = True

        blocked = int(out_body[:, 1].min()) > 1 if direction == -1 else int(out_body[:, 1].max()) < w - 2
        placed.append(
            {
                **obj,
                "out_body": out_body,
                "out_holes": out_holes,
                "out_silhouette": out_sil,
                "blocked": bool(blocked),
            }
        )
    return placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide hollow objects to the left or right wall by hole-area parity, then fill their holes with the target wall color.

    Input structure:
    - The leftmost column and rightmost column are solid walls of two different colors.
    - The interior contains several 8-connected hollow objects of one third color.
    - Each object encloses one or more background holes, while object color differs from both wall colors.

    Rule:
    - For each hollow object, sum the sizes of all enclosed holes inside that object.
    - If the total hole area is even, slide the object horizontally toward the left wall; if odd, slide it toward the right wall.
    - Within the same target wall, process objects from nearest to farthest from that wall, sliding each rigidly until its hollow silhouette would hit the wall or an already settled object.
    - After an object settles, fill all of its enclosed holes with the target wall color; keep both wall columns unchanged.

    Key insight:
    - The holes serve twice: their total area decides the travel direction, and after landing those same holes are recolored as the wall’s signature.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5, "expected a 2D grid with room for walls and hollow objects"
    bg, left_color, right_color, object_color, objects = _parse_objects(g)

    out = np.full_like(g, int(bg))
    out[:, 0] = int(left_color)
    out[:, -1] = int(right_color)

    left_objects = [obj for obj in objects if obj["target_side"] == "left"]
    right_objects = [obj for obj in objects if obj["target_side"] == "right"]
    assert left_objects and right_objects, "valid tasks must contain both parity branches"

    placed = _settle_side(g.shape, left_objects, 0, -1) + _settle_side(g.shape, right_objects, g.shape[1] - 1, 1)
    occupied = out != bg

    for obj in placed:
        assert not np.any(occupied[obj["out_body"][:, 0], obj["out_body"][:, 1]]), "object bodies must not overlap"
        out[obj["out_body"][:, 0], obj["out_body"][:, 1]] = int(object_color)
        occupied[obj["out_body"][:, 0], obj["out_body"][:, 1]] = True

        fill_color = int(left_color if obj["target_side"] == "left" else right_color)
        for hole in obj["out_holes"]:
            assert not np.any(occupied[hole[:, 0], hole[:, 1]]), "filled holes must not overlap prior output"
            out[hole[:, 0], hole[:, 1]] = fill_color
            occupied[hole[:, 0], hole[:, 1]] = True

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (18, 26)
OBJECT_COLOR_POOL = tuple(range(10))
OBJECT_COUNT_PATTERNS = ((2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2), (2, 3))
MAX_ATTEMPTS = 600


def _random_palette() -> tuple[int, int, int, int]:
    bg, left_color, right_color, object_color = random.sample(OBJECT_COLOR_POOL, 4)
    return int(bg), int(left_color), int(right_color), int(object_color)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            out[rr[ok], cc[ok]] = True
    return out


def _add_extension(mask: np.ndarray) -> np.ndarray:
    if random.random() < 0.45:
        return mask

    h, w = mask.shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side in {"top", "bottom"}:
        ext_h = random.randint(1, min(3, 9 - h))
        ext_w = random.randint(max(2, w // 2), w)
        off = random.randint(0, w - ext_w)
        out = np.zeros((h + ext_h, w), dtype=bool)
        if side == "top":
            out[ext_h:, :] = mask
            out[:ext_h, off : off + ext_w] = True
        else:
            out[:h, :] = mask
            out[h:, off : off + ext_w] = True
        return out

    ext_w = random.randint(1, min(3, 9 - w))
    ext_h = random.randint(max(2, h // 2), h)
    off = random.randint(0, h - ext_h)
    out = np.zeros((h, w + ext_w), dtype=bool)
    if side == "left":
        out[:, ext_w:] = mask
        out[off : off + ext_h, :ext_w] = True
    else:
        out[:, :w] = mask
        out[off : off + ext_h, w:] = True
    return out


def _generate_hollow_mask(parity: int) -> tuple[np.ndarray, list[CoordArray], int] | None:
    for _ in range(200):
        base_h = random.randint(5, 7)
        base_w = random.randint(5, 8)
        base = np.ones((base_h, base_w), dtype=bool)
        n_holes = 1 if random.random() < 0.65 else 2

        occupied = np.zeros((base_h, base_w), dtype=bool)
        chosen: list[tuple[int, int, int, int]] = []
        area = 0
        ok = True
        for hole_idx in range(n_holes):
            candidates: list[tuple[int, int, int, int]] = []
            for hh in range(1, min(3, base_h - 2) + 1):
                for hw in range(1, min(3, base_w - 2) + 1):
                    if hole_idx == n_holes - 1 and (area + hh * hw) % 2 != parity:
                        continue
                    for hr in range(1, base_h - hh):
                        for hc in range(1, base_w - hw):
                            halo = occupied[max(0, hr - 1) : min(base_h, hr + hh + 1), max(0, hc - 1) : min(base_w, hc + hw + 1)]
                            if np.any(halo):
                                continue
                            candidates.append((hr, hc, hh, hw))
            if not candidates:
                ok = False
                break
            hr, hc, hh, hw = random.choice(candidates)
            occupied[hr : hr + hh, hc : hc + hw] = True
            chosen.append((hr, hc, hh, hw))
            area += int(hh * hw)

        if not ok or area % 2 != parity:
            continue

        for hr, hc, hh, hw in chosen:
            base[hr : hr + hh, hc : hc + hw] = False

        mask = _add_extension(base)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)

        body_coords = np.argwhere(mask)
        holes = _holes_in_component(mask, body_coords)
        hole_area = sum(int(h.shape[0]) for h in holes)
        if not holes or hole_area % 2 != parity:
            continue
        return mask.copy(), holes, int(hole_area)
    return None


def _build_object(parity: int) -> dict | None:
    built = _generate_hollow_mask(parity)
    if built is None:
        return None
    mask, holes, hole_area = built
    body_rel = np.argwhere(mask)
    silhouette_rel = np.vstack([body_rel] + holes)
    return {
        "mask": mask,
        "body_rel": body_rel,
        "hole_rels": holes,
        "silhouette_rel": silhouette_rel,
        "shape": mask.shape,
        "hole_area": int(hole_area),
        "target_side": "left" if parity == 0 else "right",
    }


def _place_output_side(shape: tuple[int, int], objects: list[dict], side: str) -> list[dict] | None:
    h, w = shape
    direction = -1 if side == "left" else 1
    wall_col = 0 if side == "left" else w - 1
    settled = np.zeros((h, w), dtype=bool)
    settled[:, wall_col] = True
    placed: list[dict] = []

    row_candidates = list(range(h))
    for obj in objects:
        oh, _ow = obj["shape"]
        random.shuffle(row_candidates)
        found = None
        for row in row_candidates:
            if row + oh > h:
                continue
            if side == "left":
                col = 1
                while col < w - 1:
                    sil = _shift(obj["silhouette_rel"], row, col)
                    if int(sil[:, 1].max()) >= w - 1:
                        col = w
                        break
                    if np.any(settled[sil[:, 0], sil[:, 1]]):
                        col += 1
                        continue
                    found = (row, col)
                    break
            else:
                col = w - 1 - obj["shape"][1]
                while col >= 1:
                    sil = _shift(obj["silhouette_rel"], row, col)
                    if int(sil[:, 1].min()) <= 0:
                        col = -1
                        break
                    if np.any(settled[sil[:, 0], sil[:, 1]]):
                        col -= 1
                        continue
                    found = (row, col)
                    break
            if found is not None:
                break
        if found is None:
            return None

        row, col = found
        out_body = _shift(obj["body_rel"], row, col)
        out_holes = [_shift(hole_rel, row, col) for hole_rel in obj["hole_rels"]]
        out_sil = _shift(obj["silhouette_rel"], row, col)
        settled[out_sil[:, 0], out_sil[:, 1]] = True
        blocked = int(out_body[:, 1].min()) > 1 if side == "left" else int(out_body[:, 1].max()) < w - 2
        placed.append(
            {
                **obj,
                "coords": out_body.copy(),
                "bbox": _bbox(out_body),
                "out_body": out_body,
                "out_holes": out_holes,
                "out_silhouette": out_sil,
                "blocked": bool(blocked),
            }
        )
    return placed


def _render_output(shape: tuple[int, int], bg: int, left_color: int, right_color: int, object_color: int, placed: list[dict]) -> np.ndarray | None:
    out = np.full(shape, int(bg), dtype=int)
    out[:, 0] = int(left_color)
    out[:, -1] = int(right_color)
    occupied = out != bg
    for obj in placed:
        if np.any(occupied[obj["out_silhouette"][:, 0], obj["out_silhouette"][:, 1]]):
            return None
        out[obj["out_body"][:, 0], obj["out_body"][:, 1]] = int(object_color)
        fill_color = int(left_color if obj["target_side"] == "left" else right_color)
        for hole in obj["out_holes"]:
            out[hole[:, 0], hole[:, 1]] = fill_color
        occupied[obj["out_silhouette"][:, 0], obj["out_silhouette"][:, 1]] = True
    return out


def _scatter_to_input(shape: tuple[int, int], bg: int, left_color: int, right_color: int, object_color: int, placed: list[dict]) -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    h, w = shape
    inp = np.full(shape, int(bg), dtype=int)
    inp[:, 0] = int(left_color)
    inp[:, -1] = int(right_color)

    occupied_sil = np.zeros((h, w), dtype=bool)
    occupied_sil[:, 0] = True
    occupied_sil[:, -1] = True
    body_mask = np.zeros((h, w), dtype=bool)

    left_seen: set[int] = set()
    right_seen: set[int] = set()
    scattered: list[dict] = []
    order = placed[:]
    random.shuffle(order)

    for obj in order:
        side = obj["target_side"]
        bbox = _bbox(obj["out_body"])
        max_shift = (w - 2 - bbox[3]) if side == "left" else (bbox[1] - 1)
        if max_shift < 2:
            return None, None
        offsets = list(range(2, int(max_shift) + 1))
        random.shuffle(offsets)
        chosen = None
        for offset in offsets:
            distance = int(bbox[1] + offset) if side == "left" else int((w - 1) - (bbox[3] - offset))
            if side == "left" and distance in left_seen:
                continue
            if side == "right" and distance in right_seen:
                continue

            dc = offset if side == "left" else -offset
            in_body = _shift(obj["out_body"], 0, dc)
            in_holes = [_shift(hole, 0, dc) for hole in obj["out_holes"]]
            in_sil = _shift(obj["out_silhouette"], 0, dc)

            if np.any(occupied_sil[in_sil[:, 0], in_sil[:, 1]]):
                continue
            trial_mask = body_mask.copy()
            trial_mask[in_body[:, 0], in_body[:, 1]] = True
            if np.any(_dilate8(body_mask)[in_body[:, 0], in_body[:, 1]]):
                continue
            chosen = (offset, distance, in_body, in_holes, in_sil)
            break
        if chosen is None:
            return None, None

        _offset, distance, in_body, in_holes, in_sil = chosen
        inp[in_body[:, 0], in_body[:, 1]] = int(object_color)
        occupied_sil[in_sil[:, 0], in_sil[:, 1]] = True
        body_mask[in_body[:, 0], in_body[:, 1]] = True
        if side == "left":
            left_seen.add(int(distance))
        else:
            right_seen.add(int(distance))
        scattered.append(
            {
                **obj,
                "coords": in_body.copy(),
                "holes": in_holes,
                "bbox": _bbox(in_body),
                "in_body": in_body,
                "in_holes": in_holes,
                "in_silhouette": in_sil,
                "distance": int(distance),
            }
        )
    return inp, scattered


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if np.array_equal(inp, expected):
        return False
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    try:
        bg, left_color, right_color, object_color, parsed = _parse_objects(inp)
    except AssertionError:
        return False
    if (bg, left_color, right_color, object_color) != (
        meta["bg"],
        meta["left_color"],
        meta["right_color"],
        meta["object_color"],
    ):
        return False

    if len(parsed) != len(meta["objects"]):
        return False
    if not any(obj["hole_area"] % 2 == 0 for obj in parsed):
        return False
    if not any(obj["hole_area"] % 2 == 1 for obj in parsed):
        return False

    blocked_ok = False
    for side in ("left", "right"):
        side_objs = [obj for obj in meta["objects"] if obj["target_side"] == side]
        distances = [int(obj["distance"]) for obj in side_objs]
        if len(distances) != len(set(distances)):
            return False
        if len(side_objs) >= 2 and any(obj["blocked"] for obj in side_objs):
            blocked_ok = True
    if not blocked_ok:
        return False

    for obj in meta["objects"]:
        if np.any(inp[obj["in_body"][:, 0], obj["in_body"][:, 1]] != meta["object_color"]):
            return False
        for hole in obj["in_holes"]:
            if np.any(inp[hole[:, 0], hole[:, 1]] != meta["bg"]):
                return False
        fill_color = meta["left_color"] if obj["target_side"] == "left" else meta["right_color"]
        for hole in obj["out_holes"]:
            if np.any(expected[hole[:, 0], hole[:, 1]] != fill_color):
                return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | tuple[None, None, None]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg, left_color, right_color, object_color = _random_palette()
    left_count, right_count = random.choice(OBJECT_COUNT_PATTERNS)

    left_objects = []
    for _ in range(left_count):
        obj = _build_object(parity=0)
        if obj is None:
            return None, None, None
        left_objects.append(obj)

    right_objects = []
    for _ in range(right_count):
        obj = _build_object(parity=1)
        if obj is None:
            return None, None, None
        right_objects.append(obj)

    left_placed = _place_output_side((h, w), left_objects, "left")
    right_placed = _place_output_side((h, w), right_objects, "right")
    if left_placed is None or right_placed is None:
        return None, None, None

    if left_count >= 2 and not any(obj["blocked"] for obj in left_placed):
        return None, None, None
    if right_count >= 2 and not any(obj["blocked"] for obj in right_placed):
        return None, None, None

    all_placed = left_placed + right_placed
    expected = _render_output((h, w), bg, left_color, right_color, object_color, all_placed)
    if expected is None:
        return None, None, None

    inp, scattered = _scatter_to_input((h, w), bg, left_color, right_color, object_color, all_placed)
    if inp is None or scattered is None:
        return None, None, None

    meta = {
        "bg": int(bg),
        "left_color": int(left_color),
        "right_color": int(right_color),
        "object_color": int(object_color),
        "objects": scattered,
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp, expected, meta = _build_case()
        if inp is None or expected is None or meta is None:
            continue
        if _validate_case(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid parity-guided hollow-object wall-stacking input")

from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
MOTIF_COLOR = 1
ANCHOR_COLORS = [2, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")

BASE_MOTIFS = [
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=sprite.dtype))


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = 0) -> None:
    h, w = dst.shape
    ph, pw = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r0) + ph)
    cc1 = min(w, int(c0) + pw)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = view != int(transparent)
    dst[rr0:rr1, cc0:cc1][mask] = view[mask]


def _components_by_color(grid: np.ndarray, *, bg: int = 0) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_solid_square(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    if side_r != side_c or int(comp["area"]) != side_r * side_c:
        return False
    patch = _crop_bbox(grid, comp["bbox"])
    return bool(np.all(patch == int(comp["color"])))


def _classify_direction(
    anchor_bbox: tuple[int, int, int, int],
    motif_bbox: tuple[int, int, int, int],
) -> str:
    ar0, ac0, ar1, ac1 = anchor_bbox
    mr0, mc0, mr1, mc1 = motif_bbox
    center_r = (mr0 + mr1) // 2
    center_c = (mc0 + mc1) // 2

    if ac1 < mc0 and ar0 <= center_r <= ar1:
        return "left"
    if ac0 > mc1 and ar0 <= center_r <= ar1:
        return "right"
    if ar1 < mr0 and ac0 <= center_c <= ac1:
        return "up"
    if ar0 > mr1 and ac0 <= center_c <= ac1:
        return "down"

    anchor_center_r = (ar0 + ar1) / 2.0
    anchor_center_c = (ac0 + ac1) / 2.0
    motif_center_r = (mr0 + mr1) / 2.0
    motif_center_c = (mc0 + mc1) / 2.0
    if abs(anchor_center_c - motif_center_c) >= abs(anchor_center_r - motif_center_r):
        return "left" if anchor_center_c < motif_center_c else "right"
    return "up" if anchor_center_r < motif_center_r else "down"


def _extract_layout(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[dict]]:
    comps = _components_by_color(grid, bg=BG)
    motif_comps = [comp for comp in comps if int(comp["color"]) == MOTIF_COLOR]
    assert motif_comps, "expected one blue motif component"
    motif = max(motif_comps, key=lambda comp: int(comp["area"]))
    motif_bbox = motif["bbox"]
    motif_patch = _crop_bbox(grid, motif_bbox)

    anchors = []
    seen_dirs = set()
    for comp in comps:
        if int(comp["color"]) == MOTIF_COLOR:
            continue
        if not _is_solid_square(grid, comp):
            continue
        direction = _classify_direction(comp["bbox"], motif_bbox)
        assert direction not in seen_dirs, "each cardinal direction may contain at most one anchor"
        seen_dirs.add(direction)
        side = comp["bbox"][2] - comp["bbox"][0] + 1
        anchors.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "size": int(side),
                "direction": direction,
            }
        )
    assert 3 <= len(anchors) <= 4, "expected three or four anchors"
    return motif_patch, motif_bbox, sorted(anchors, key=lambda item: DIRECTIONS.index(item["direction"]))


def _stamp_origin(anchor_bbox: tuple[int, int, int, int], direction: str, scale: int) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = anchor_bbox
    k = int(scale)
    if direction == "right":
        return ar0 - k, ac1 + 1
    if direction == "left":
        return ar0 - k, ac0 - 3 * k
    if direction == "down":
        return ar1 + 1, ac0 - k
    if direction == "up":
        return ar0 - 3 * k, ac0 - k
    raise ValueError(f"unknown direction: {direction}")


def _transformed_patch(motif_patch: np.ndarray, direction: str, color: int, scale: int) -> np.ndarray:
    mask = (motif_patch == MOTIF_COLOR).astype(int)
    scaled = _scale_sprite(mask, int(scale))
    if direction in {"left", "right"}:
        scaled = np.fliplr(scaled)
    else:
        scaled = np.flipud(scaled)
    return np.where(scaled == 1, int(color), BG).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored scaled copies of the blue motif outside each square anchor.

    Input structure:
    - Black (0) background with one small blue (1) asymmetric motif near the center.
    - Three or four other monochrome solid square blocks of distinct non-blue colors.
    - Each square anchor lies strictly above, below, left, or right of the motif, with at most one anchor per direction.

    Rule:
    - Find the blue motif and its bounding-box center.
    - For each colored square anchor, read its cardinal direction relative to the motif and its side length k.
    - Scale the motif by factor k with nearest-neighbor replication, flip it horizontally for left/right anchors or vertically for up/down anchors, recolor the non-black pixels to the anchor color, and stamp the result immediately outside that anchor on its outer side.
    - Keep the original motif and anchors unchanged; clip any stamped pixels that would leave the grid.

    Key insight:
    - Each anchor encodes two independent parameters at once: direction chooses the reflection axis, while square size chooses the scale factor.
    """
    g = np.asarray(grid, dtype=int)
    motif_patch, _motif_bbox, anchors = _extract_layout(g)

    out = g.copy()
    for anchor in anchors:
        patch = _transformed_patch(motif_patch, anchor["direction"], anchor["color"], anchor["size"])
        r0, c0 = _stamp_origin(anchor["bbox"], anchor["direction"], anchor["size"])
        _paste_clip(out, patch, r0, c0, transparent=BG)
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], *, pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _bbox_from_origin(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _sample_motif() -> np.ndarray:
    for _ in range(50):
        motif = random.choice(BASE_MOTIFS).copy()
        motif = np.rot90(motif, random.randrange(4))
        if motif.shape != (3, 3):
            continue
        if np.array_equal(motif, np.fliplr(motif)):
            continue
        if np.array_equal(motif, np.flipud(motif)):
            continue
        return motif
    raise RuntimeError("failed to sample a non-degenerate motif")


def _sample_directions() -> list[str]:
    if random.random() < 0.55:
        return list(DIRECTIONS)
    missing = random.choice(DIRECTIONS)
    return [d for d in DIRECTIONS if d != missing]


def _sample_sizes(directions: list[str]) -> dict[str, int] | None:
    for _ in range(200):
        sizes = {direction: random.choices([1, 2, 3], weights=[5, 4, 3])[0] for direction in directions}
        if len(set(sizes.values())) < 2:
            continue
        if max(sizes.values()) == 1:
            continue
        if "left" in sizes and "right" in sizes and sizes["left"] + sizes["right"] > 3:
            continue
        if "up" in sizes and "down" in sizes and sizes["up"] + sizes["down"] > 3:
            continue
        return sizes
    return None


def _layout_candidates(
    direction: str,
    size: int,
    motif_r0: int,
    motif_c0: int,
    h: int,
    w: int,
) -> list[dict]:
    center_r = motif_r0 + 1
    center_c = motif_c0 + 1
    gap = 3 * int(size)
    candidates = []

    if direction == "left":
        anchor_c0 = motif_c0 - gap - size
        for anchor_r0 in range(center_r - size + 1, center_r + 1):
            anchor_bbox = _bbox_from_origin(anchor_r0, anchor_c0, size, size)
            stamp_bbox = _bbox_from_origin(anchor_r0 - size, anchor_c0 - 3 * size, 3 * size, 3 * size)
            if stamp_bbox[0] < 0 or anchor_bbox[0] < 0 or stamp_bbox[1] < 0:
                continue
            if anchor_bbox[2] >= h or stamp_bbox[2] >= h:
                continue
            candidates.append({"anchor_bbox": anchor_bbox, "stamp_bbox": stamp_bbox})

    elif direction == "right":
        anchor_c0 = motif_c0 + 3 + gap
        for anchor_r0 in range(center_r - size + 1, center_r + 1):
            anchor_bbox = _bbox_from_origin(anchor_r0, anchor_c0, size, size)
            stamp_bbox = _bbox_from_origin(anchor_r0 - size, anchor_c0 + size, 3 * size, 3 * size)
            if stamp_bbox[0] < 0 or anchor_bbox[0] < 0:
                continue
            if anchor_bbox[2] >= h or stamp_bbox[2] >= h or stamp_bbox[3] >= w:
                continue
            candidates.append({"anchor_bbox": anchor_bbox, "stamp_bbox": stamp_bbox})

    elif direction == "up":
        anchor_r0 = motif_r0 - gap - size
        for anchor_c0 in range(center_c - size + 1, center_c + 1):
            anchor_bbox = _bbox_from_origin(anchor_r0, anchor_c0, size, size)
            stamp_bbox = _bbox_from_origin(anchor_r0 - 3 * size, anchor_c0 - size, 3 * size, 3 * size)
            if stamp_bbox[0] < 0 or anchor_bbox[1] < 0 or stamp_bbox[1] < 0:
                continue
            if anchor_bbox[3] >= w or stamp_bbox[3] >= w:
                continue
            candidates.append({"anchor_bbox": anchor_bbox, "stamp_bbox": stamp_bbox})

    elif direction == "down":
        anchor_r0 = motif_r0 + 3 + gap
        for anchor_c0 in range(center_c - size + 1, center_c + 1):
            anchor_bbox = _bbox_from_origin(anchor_r0, anchor_c0, size, size)
            stamp_bbox = _bbox_from_origin(anchor_r0 + size, anchor_c0 - size, 3 * size, 3 * size)
            if stamp_bbox[1] < 0 or anchor_bbox[1] < 0:
                continue
            if anchor_bbox[3] >= w or stamp_bbox[3] >= w or stamp_bbox[2] >= h:
                continue
            candidates.append({"anchor_bbox": anchor_bbox, "stamp_bbox": stamp_bbox})
    random.shuffle(candidates)
    return candidates


def _assign_positions(
    motif_bbox: tuple[int, int, int, int],
    sizes: dict[str, int],
    h: int,
    w: int,
) -> dict[str, dict] | None:
    mr0, mc0, _mr1, _mc1 = motif_bbox
    options = {
        direction: _layout_candidates(direction, sizes[direction], mr0, mc0, h, w)
        for direction in sizes
    }
    if any(not values for values in options.values()):
        return None

    placed: dict[str, dict] = {}
    boxes: list[tuple[str, tuple[int, int, int, int]]] = [("motif", motif_bbox)]
    order = sorted(sizes, key=lambda direction: -sizes[direction])

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        direction = order[idx]
        for candidate in options[direction]:
            anchor_bbox = candidate["anchor_bbox"]
            stamp_bbox = candidate["stamp_bbox"]
            conflict = False
            for name, bbox in boxes:
                if name == f"stamp:{direction}" or name == f"anchor:{direction}":
                    continue
                if _boxes_overlap(anchor_bbox, bbox, pad=1) or _boxes_overlap(stamp_bbox, bbox, pad=1):
                    conflict = True
                    break
            if conflict or _boxes_overlap(anchor_bbox, stamp_bbox, pad=0):
                continue
            boxes.append((f"anchor:{direction}", anchor_bbox))
            boxes.append((f"stamp:{direction}", stamp_bbox))
            placed[direction] = candidate
            if backtrack(idx + 1):
                return True
            boxes.pop()
            boxes.pop()
            placed.pop(direction, None)
        return False

    return placed if backtrack(0) else None


def _render_input(spec: dict) -> np.ndarray:
    grid = np.zeros((spec["h"], spec["w"]), dtype=int)
    mr0, mc0, mr1, mc1 = spec["motif_bbox"]
    motif = spec["motif"]
    grid[mr0 : mr1 + 1, mc0 : mc1 + 1][motif == 1] = MOTIF_COLOR
    for anchor in spec["anchors"]:
        r0, c0, r1, c1 = anchor["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(anchor["color"])
    return grid


def _expected_output(spec: dict) -> np.ndarray:
    out = _render_input(spec)
    motif_patch = np.where(spec["motif"] == 1, MOTIF_COLOR, BG)
    for anchor in spec["anchors"]:
        patch = _transformed_patch(motif_patch, anchor["direction"], anchor["color"], anchor["size"])
        r0, c0 = _stamp_origin(anchor["bbox"], anchor["direction"], anchor["size"])
        _paste_clip(out, patch, r0, c0, transparent=BG)
    return out


def _build_spec() -> dict | None:
    directions = _sample_directions()
    sizes = _sample_sizes(directions)
    if sizes is None:
        return None

    req_top = 7 * sizes.get("up", 0)
    req_bottom = 7 * sizes.get("down", 0)
    req_left = 7 * sizes.get("left", 0)
    req_right = 7 * sizes.get("right", 0)

    min_h = max(23, req_top + 3 + req_bottom)
    min_w = max(23, req_left + 3 + req_right)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    motif_r0 = random.randint(req_top, h - 3 - req_bottom)
    motif_c0 = random.randint(req_left, w - 3 - req_right)
    motif_bbox = _bbox_from_origin(motif_r0, motif_c0, 3, 3)

    positions = _assign_positions(motif_bbox, sizes, h, w)
    if positions is None:
        return None

    colors = random.sample(ANCHOR_COLORS, len(directions))
    motif = _sample_motif()
    anchors = []
    for direction, color in zip(sorted(directions, key=DIRECTIONS.index), colors, strict=False):
        anchor_bbox = positions[direction]["anchor_bbox"]
        anchors.append(
            {
                "direction": direction,
                "size": int(sizes[direction]),
                "bbox": anchor_bbox,
                "color": int(color),
            }
        )

    return {
        "h": int(h),
        "w": int(w),
        "motif": motif,
        "motif_bbox": motif_bbox,
        "anchors": anchors,
    }


def _semantic_validate(spec: dict) -> bool:
    inp = _render_input(spec)
    out = _expected_output(spec)

    if not np.array_equal(main(inp), out):
        return False

    motif_patch, motif_bbox, anchors = _extract_layout(inp)
    if motif_bbox != spec["motif_bbox"]:
        return False
    if not np.array_equal((motif_patch == MOTIF_COLOR).astype(int), spec["motif"]):
        return False

    got = {anchor["direction"]: anchor for anchor in anchors}
    want = {anchor["direction"]: anchor for anchor in spec["anchors"]}
    if set(got) != set(want):
        return False

    if not ({"left", "right"} & set(got)) or not ({"up", "down"} & set(got)):
        return False
    if len({anchor["size"] for anchor in anchors}) < 2:
        return False

    for direction in got:
        if got[direction]["size"] != want[direction]["size"]:
            return False
        if got[direction]["color"] != want[direction]["color"]:
            return False
        if got[direction]["bbox"] != want[direction]["bbox"]:
            return False

        correct = _transformed_patch(motif_patch, direction, got[direction]["color"], got[direction]["size"])
        opposite = _transformed_patch(
            motif_patch,
            "left" if direction in {"up", "down"} else "up",
            got[direction]["color"],
            got[direction]["size"],
        )
        if np.array_equal(correct, opposite):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        spec = _build_spec()
        if spec is None:
            continue
        if _semantic_validate(spec):
            return _render_input(spec)
    raise RuntimeError("failed to generate a valid input")

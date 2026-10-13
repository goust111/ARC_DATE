from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 500
ARC_MAX = 30


@dataclass
class HoleSpec:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    key: tuple[tuple[int, int], bytes]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _component_patch(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = np.zeros(patch.shape, dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
        "area": int(mask.sum()),
    }


def _inner_holes(mask: np.ndarray) -> list[HoleSpec]:
    h, w = mask.shape
    holes: list[HoleSpec] = []
    for coords in _components(~mask):
        touches_edge = bool(
            (coords[:, 0] == 0).any()
            or (coords[:, 0] == h - 1).any()
            or (coords[:, 1] == 0).any()
            or (coords[:, 1] == w - 1).any()
        )
        if touches_edge:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        hole_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        hole_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        holes.append(HoleSpec((r0, c0, r1, c1), hole_mask, _shape_key(hole_mask)))
    return holes


def _crop_non_bg(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    assert coords.size > 0, "completed template must contain colored cells"
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _pack_completed(items: list[dict]) -> np.ndarray:
    items = sorted(items, key=lambda item: (int(item["area"]), int(item["left"]), int(item["top"])))
    height = max(int(item["patch"].shape[0]) for item in items)
    width = sum(int(item["patch"].shape[1]) for item in items) + len(items) - 1
    out = np.full((height, width), BG, dtype=int)
    c = 0
    for item in items:
        patch = item["patch"]
        h, w = map(int, patch.shape)
        r = height - h
        out[r : r + h, c : c + w] = patch
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered colored fragments back into matching holes and pack the completed templates by area.

    Input structure:
    - The background is black.
    - Several large monochrome connected template objects contain one or more internal black rectangular holes.
    - Separate small colored connected fragments are scattered elsewhere; each fragment has exactly the same binary shape as one template hole, while its color is only a distractor.

    Rule:
    - Find every internal background hole in each template and match it to the unique fragment with the same translated binary shape.
    - Fill that hole with the matched fragment, preserving the fragment's own color pixels, then crop each completed template to its tight non-background bbox.
    - Sort the completed templates by their total non-background area from smallest to largest and place them left-to-right on a new black canvas with one black separator column, bottom aligned.

    Key insight:
    The fragments are assigned by shape rather than by color, and the final left-to-right order is determined by completed object area rather than by where the templates appeared in the input.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates: list[dict] = []
    pieces: list[dict] = []
    for coords in _components(g != BG):
        obj = _component_patch(g, coords)
        holes = _inner_holes(obj["mask"])
        if holes:
            obj["holes"] = holes
            templates.append(obj)
        else:
            obj["key"] = _shape_key(obj["mask"])
            pieces.append(obj)

    assert templates, "expected at least one holed template"
    assert pieces, "expected at least one detached fragment"

    pieces_by_shape: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for i, piece in enumerate(pieces):
        pieces_by_shape.setdefault(piece["key"], []).append(i)

    used_pieces: set[int] = set()
    completed: list[dict] = []
    for tmpl in templates:
        patch = tmpl["patch"].copy()
        for hole in tmpl["holes"]:
            candidates = [i for i in pieces_by_shape.get(hole.key, []) if i not in used_pieces]
            assert len(candidates) == 1, "each hole shape must match exactly one unused fragment"
            piece = pieces[candidates[0]]
            assert np.array_equal(piece["mask"], hole.mask), "fragment and hole masks must match"

            hr0, hc0, hr1, hc1 = hole.bbox
            view = patch[hr0 : hr1 + 1, hc0 : hc1 + 1]
            view[hole.mask] = piece["patch"][piece["mask"]]
            used_pieces.add(candidates[0])

        cropped = _crop_non_bg(patch)
        tr0, tc0, _tr1, _tc1 = tmpl["bbox"]
        completed.append(
            {
                "patch": cropped,
                "area": int(np.count_nonzero(cropped != BG)),
                "top": int(tr0),
                "left": int(tc0),
            }
        )

    assert len(used_pieces) == len(pieces), "every fragment must be used by one hole"
    return _pack_completed(completed)


# --------------- generator ---------------


def _rect_hole_key(h: int, w: int) -> tuple[tuple[int, int], bytes]:
    return _shape_key(np.ones((int(h), int(w)), dtype=bool))


def _sample_dimensions(n_templates: int) -> list[tuple[int, int]] | None:
    max_w_by_count = {2: 10, 3: 8, 4: 6}
    max_h_by_count = {2: 10, 3: 9, 4: 8}
    for _ in range(200):
        dims: list[tuple[int, int]] = []
        used_areas: set[int] = set()
        for _i in range(n_templates):
            choices = [
                (h, w)
                for h in range(5, max_h_by_count[n_templates] + 1)
                for w in range(5, max_w_by_count[n_templates] + 1)
                if h * w not in used_areas
            ]
            if not choices:
                break
            h, w = random.choice(choices)
            dims.append((h, w))
            used_areas.add(h * w)
        if len(dims) == n_templates and sum(w for _h, w in dims) + n_templates - 1 <= ARC_MAX:
            return dims
    return None


def _hole_count_for(h: int, w: int) -> int:
    if h * w >= 70:
        return random.choice([1, 2, 3])
    if h * w >= 42:
        return random.choice([1, 2])
    return 1


def _make_template_patch(
    h: int,
    w: int,
    color: int,
    n_holes: int,
    used_keys: set[tuple[tuple[int, int], bytes]],
) -> tuple[np.ndarray, list[HoleSpec]] | None:
    for _attempt in range(100):
        patch = np.full((h, w), int(color), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        holes: list[HoleSpec] = []
        local_keys: set[tuple[tuple[int, int], bytes]] = set()

        for _ in range(n_holes):
            candidates: list[tuple[int, int, int, int, tuple[tuple[int, int], bytes]]] = []
            for hh in range(1, min(4, h - 2) + 1):
                for ww in range(1, min(4, w - 2) + 1):
                    area = hh * ww
                    if area < 2 or area > 6:
                        continue
                    key = _rect_hole_key(hh, ww)
                    if key in used_keys or key in local_keys:
                        continue
                    for r0 in range(1, h - hh):
                        for c0 in range(1, w - ww):
                            rr0 = max(0, r0 - 1)
                            cc0 = max(0, c0 - 1)
                            rr1 = min(h, r0 + hh + 1)
                            cc1 = min(w, c0 + ww + 1)
                            if not reserved[rr0:rr1, cc0:cc1].any():
                                candidates.append((r0, c0, hh, ww, key))
            if not candidates:
                break

            r0, c0, hh, ww, key = random.choice(candidates)
            patch[r0 : r0 + hh, c0 : c0 + ww] = BG
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + hh + 1)
            cc1 = min(w, c0 + ww + 1)
            reserved[rr0:rr1, cc0:cc1] = True

            hole_mask = np.ones((hh, ww), dtype=bool)
            holes.append(HoleSpec((r0, c0, r0 + hh - 1, c0 + ww - 1), hole_mask, key))
            local_keys.add(key)
        else:
            used_keys.update(local_keys)
            return patch, holes
    return None


def _complete_template(template_patch: np.ndarray, holes: list[HoleSpec], piece_patches: list[np.ndarray]) -> np.ndarray:
    out = template_patch.copy()
    for hole, piece in zip(holes, piece_patches):
        r0, c0, r1, c1 = hole.bbox
        out[r0 : r1 + 1, c0 : c1 + 1][hole.mask] = piece[piece != BG]
    return out


def _place_items(items: list[dict], height: int, width: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int]] = [(-1, -1)] * len(items)

    order = list(range(len(items)))
    random.shuffle(order)
    order.sort(key=lambda i: int(items[i]["patch"].shape[0] * items[i]["patch"].shape[1]), reverse=True)

    for idx in order:
        patch = items[idx]["patch"]
        h, w = map(int, patch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, height - h + 1):
            for c0 in range(0, width - w + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(height, r0 + h + 1)
                cc1 = min(width, c0 + w + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        mask = patch != BG
        view = grid[r0 : r0 + h, c0 : c0 + w]
        view[mask] = patch[mask]

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(height, r0 + h + 1)
        cc1 = min(width, c0 + w + 1)
        blocked[rr0:rr1, cc0:cc1] = True
        positions[idx] = (r0, c0)

    return grid, positions


def _gen_attempt() -> np.ndarray | None:
    n_templates = random.randint(2, 4)
    dims = _sample_dimensions(n_templates)
    if dims is None:
        return None

    template_colors = random.sample(COLORS, n_templates)
    fragment_colors = [c for c in COLORS if c not in template_colors]
    if not fragment_colors:
        return None

    used_hole_keys: set[tuple[tuple[int, int], bytes]] = set()
    templates: list[dict] = []
    pieces: list[dict] = []
    items: list[dict] = []

    for index, ((h, w), color) in enumerate(zip(dims, template_colors)):
        n_holes = _hole_count_for(h, w)
        made = _make_template_patch(h, w, color, n_holes, used_hole_keys)
        if made is None:
            return None
        template_patch, holes = made

        piece_patches: list[np.ndarray] = []
        for hole in holes:
            ph, pw = map(int, hole.mask.shape)
            piece_color = random.choice(fragment_colors)
            piece = np.full((ph, pw), int(piece_color), dtype=int)
            piece_patches.append(piece)
            pieces.append({"patch": piece, "template_index": index, "hole": hole})

        completed = _complete_template(template_patch, holes, piece_patches)
        templates.append(
            {
                "patch": template_patch,
                "completed": completed,
                "area": int(np.count_nonzero(completed != BG)),
                "item_index": len(items),
            }
        )
        items.append({"patch": template_patch, "kind": "template", "template_index": index})

    if len({int(t["area"]) for t in templates}) != len(templates):
        return None

    random.shuffle(pieces)
    for piece in pieces:
        items.append({"patch": piece["patch"], "kind": "piece"})

    height = random.randint(22, ARC_MAX)
    width = random.randint(22, ARC_MAX)
    placed = _place_items(items, height, width)
    if placed is None:
        return None
    grid, positions = placed

    left_order = sorted(
        range(n_templates),
        key=lambda i: (positions[templates[i]["item_index"]][1], positions[templates[i]["item_index"]][0]),
    )
    area_order = sorted(range(n_templates), key=lambda i: templates[i]["area"])
    if left_order == area_order:
        return None

    expected_items = []
    for i, template in enumerate(templates):
        r0, c0 = positions[template["item_index"]]
        expected_items.append(
            {
                "patch": template["completed"],
                "area": int(template["area"]),
                "top": int(r0),
                "left": int(c0),
            }
        )
    expected = _pack_completed(expected_items)
    if max(expected.shape) > ARC_MAX:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragment-template packing input")

from __future__ import annotations

import random

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 300
ANCHOR_SIZE = 6


def _components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            parts.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return parts


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_template_mask(component: dict[str, object]) -> np.ndarray | None:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]  # type: ignore[assignment]
    if int(r1 - r0 + 1) != 3 or int(c1 - c0 + 1) != 3:
        return None

    mask = np.zeros((3, 3), dtype=bool)
    mask[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True
    if bool(mask[1, 1]):
        return None
    if int(mask.sum()) < 4:
        return None
    if not bool(mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
        return None
    return mask


def _pick_vertical(local_row: int, mask: np.ndarray) -> str:
    if local_row == 0:
        return "top"
    if local_row == 2:
        return "bottom"
    return "top" if int(mask[0, :].sum()) >= int(mask[2, :].sum()) else "bottom"


def _pick_horizontal(local_col: int, mask: np.ndarray) -> str:
    if local_col == 0:
        return "left"
    if local_col == 2:
        return "right"
    return "left" if int(mask[:, 0].sum()) >= int(mask[:, 2].sum()) else "right"


def _infer_anchor(seed: tuple[int, int], bbox: tuple[int, int, int, int], mask: np.ndarray) -> str:
    sr, sc = seed
    r0, c0, r1, c1 = bbox

    contacts: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = sr + dr, sc + dc
        if r0 <= nr <= r1 and c0 <= nc <= c1 and bool(mask[nr - r0, nc - c0]):
            contacts.append((nr - r0, nc - c0))
    assert contacts, "seed must touch its template object"

    tr, tc = contacts[0]
    if sc == c0 - 1:
        return "TR" if _pick_vertical(int(tr), mask) == "top" else "BR"
    if sc == c1 + 1:
        return "TL" if _pick_vertical(int(tr), mask) == "top" else "BL"
    if sr == r0 - 1:
        return "BL" if _pick_horizontal(int(tc), mask) == "left" else "BR"
    if sr == r1 + 1:
        return "TL" if _pick_horizontal(int(tc), mask) == "left" else "TR"
    raise AssertionError("seed must be 4-adjacent outside the template bbox")


def _top_left_from_anchor(seed: tuple[int, int], anchor: str) -> tuple[int, int]:
    sr, sc = seed
    if anchor == "TL":
        return sr, sc
    if anchor == "TR":
        return sr, sc - (ANCHOR_SIZE - 1)
    if anchor == "BL":
        return sr - (ANCHOR_SIZE - 1), sc
    if anchor == "BR":
        return sr - (ANCHOR_SIZE - 1), sc - (ANCHOR_SIZE - 1)
    raise ValueError(f"unknown anchor {anchor}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each seed pixel with a 2x enlarged copy of its adjacent 3x3 outline template.

    The enlarged copy uses the seed color rather than the template color, and the
    seed cell becomes the output corner closest to the original template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    components = _components(g, bg=BG)

    templates: dict[int, dict[str, object]] = {}
    cell_to_template: dict[tuple[int, int], int] = {}
    seeds: list[dict[str, object]] = []

    for idx, component in enumerate(components):
        coords = np.asarray(component["coords"], dtype=int)
        if int(coords.shape[0]) == 1:
            seeds.append(component)
            continue

        mask = _extract_template_mask(component)
        if mask is None:
            continue
        obj = dict(component)
        obj["mask"] = mask
        templates[idx] = obj
        for r, c in coords:
            cell_to_template[(int(r), int(c))] = idx

    out = np.full((h, w), BG, dtype=int)
    paired = 0

    for seed_comp in seeds:
        coords = np.asarray(seed_comp["coords"], dtype=int)
        sr, sc = map(int, coords[0])
        seed = (sr, sc)
        seed_color = int(seed_comp["color"])

        hits = {
            cell_to_template[(nr, nc)]
            for dr, dc in DIR4
            for nr, nc in [(sr + dr, sc + dc)]
            if (nr, nc) in cell_to_template
        }
        if len(hits) != 1:
            continue

        template = templates[next(iter(hits))]
        bbox = template["bbox"]  # type: ignore[assignment]
        mask = np.asarray(template["mask"], dtype=bool)
        anchor = _infer_anchor(seed, bbox, mask)
        top, left = _top_left_from_anchor(seed, anchor)

        expanded = np.kron(mask.astype(int), np.ones((2, 2), dtype=int)).astype(bool)
        for rr, cc in np.argwhere(expanded):
            pr = top + int(rr)
            pc = left + int(cc)
            if 0 <= pr < h and 0 <= pc < w:
                out[pr, pc] = seed_color

        paired += 1

    assert paired > 0, "expected at least one seed/template pair"
    return out


def _enumerate_templates() -> tuple[np.ndarray, ...]:
    border = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2))
    templates: list[np.ndarray] = []
    for bits in range(1 << len(border)):
        if bits.bit_count() < 5:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for i, (r, c) in enumerate(border):
            if (bits >> i) & 1:
                mask[r, c] = True
        if not bool(mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if not _is_connected(mask):
            continue
        templates.append(mask)

    templates.sort(key=lambda m: (int(m.sum()), tuple(int(v) for v in m.ravel())))
    return tuple(templates)


TEMPLATES = _enumerate_templates()
TEMPLATES_BY_CORNER = {
    (0, 0): tuple(mask for mask in TEMPLATES if bool(mask[0, 0])),
    (0, 2): tuple(mask for mask in TEMPLATES if bool(mask[0, 2])),
    (2, 0): tuple(mask for mask in TEMPLATES if bool(mask[2, 0])),
    (2, 2): tuple(mask for mask in TEMPLATES if bool(mask[2, 2])),
}

PLACEMENTS = (
    {"anchor": "TL", "seed_offset": (0, 3), "touch": (0, 2)},
    {"anchor": "TL", "seed_offset": (3, 0), "touch": (2, 0)},
    {"anchor": "TR", "seed_offset": (0, -1), "touch": (0, 0)},
    {"anchor": "TR", "seed_offset": (3, 2), "touch": (2, 2)},
    {"anchor": "BL", "seed_offset": (2, 3), "touch": (2, 2)},
    {"anchor": "BL", "seed_offset": (-1, 0), "touch": (0, 0)},
    {"anchor": "BR", "seed_offset": (2, -1), "touch": (2, 0)},
    {"anchor": "BR", "seed_offset": (-1, 2), "touch": (0, 2)},
)


def _reserve_union(
    reserved: np.ndarray,
    input_bbox: tuple[int, int, int, int],
    output_bbox: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    h, w = reserved.shape
    r0 = max(0, min(int(input_bbox[0]), int(output_bbox[0])) - pad)
    c0 = max(0, min(int(input_bbox[1]), int(output_bbox[1])) - pad)
    r1 = min(h - 1, max(int(input_bbox[2]), int(output_bbox[2])) + pad)
    c1 = min(w - 1, max(int(input_bbox[3]), int(output_bbox[3])) + pad)
    if bool(reserved[r0 : r1 + 1, c0 : c1 + 1].any()):
        return False
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _placement_output_bbox(seed: tuple[int, int], anchor: str) -> tuple[int, int, int, int]:
    r0, c0 = _top_left_from_anchor(seed, anchor)
    return r0, c0, r0 + ANCHOR_SIZE - 1, c0 + ANCHOR_SIZE - 1


def _draw_template(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = color


def _place_pair(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    object_color: int,
    seed_color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        placement = rng.choice(PLACEMENTS)
        touch = placement["touch"]
        mask = rng.choice(TEMPLATES_BY_CORNER[touch])

        obj_top = rng.randrange(h - 2)
        obj_left = rng.randrange(w - 2)
        seed = (
            obj_top + int(placement["seed_offset"][0]),
            obj_left + int(placement["seed_offset"][1]),
        )
        sr, sc = seed
        if not (0 <= sr < h and 0 <= sc < w):
            continue

        output_bbox = _placement_output_bbox(seed, str(placement["anchor"]))
        or0, oc0, or1, oc1 = output_bbox
        if not (0 <= or0 and 0 <= oc0 and or1 < h and oc1 < w):
            continue

        input_bbox = (obj_top, obj_left, obj_top + 2, obj_left + 2)
        if not _reserve_union(reserved, input_bbox, output_bbox):
            continue

        _draw_template(grid, obj_top, obj_left, mask, object_color)
        grid[sr, sc] = seed_color
        return True

    return False


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair_count = rng.randint(2, 4)
        height = rng.randint(12 + 3 * pair_count, 18 + 3 * pair_count)
        width = rng.randint(12 + 3 * pair_count, 18 + 3 * pair_count)

        palette = rng.sample(list(range(1, 10)), 2 * pair_count)
        object_colors = palette[:pair_count]
        seed_colors = palette[pair_count:]

        grid = np.full((height, width), BG, dtype=int)
        reserved = np.zeros((height, width), dtype=bool)

        ok = True
        for object_color, seed_color in zip(object_colors, seed_colors):
            if not _place_pair(rng, grid, reserved, int(object_color), int(seed_color)):
                ok = False
                break

        if ok:
            return grid

    raise RuntimeError("failed to generate a valid input")

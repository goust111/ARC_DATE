import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return (tuple(int(x) for x in m.shape), m.tobytes())


def _canonical_rot_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    masks = [_mask_key(np.rot90(mask, k)) for k in range(4)]
    return min(masks, key=lambda item: (item[0][0], item[0][1], item[1]))


def _find_template(grid: np.ndarray, bg: int) -> dict:
    comps = []
    for color in np.unique(grid):
        if int(color) == int(bg):
            continue
        for coords in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )
    assert comps, "expected at least one non-background component"
    return max(comps, key=lambda comp: (comp["area"], -comp["bbox"][1], -comp["bbox"][0]))


def _extract_holes(grid: np.ndarray, template: dict, bg: int) -> list[dict]:
    h, w = grid.shape
    template_mask = np.zeros((h, w), dtype=bool)
    template_mask[template["coords"][:, 0], template["coords"][:, 1]] = True
    r0, c0, r1, c1 = template["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for local_coords in _components(sub == int(bg)):
        if np.any(local_coords[:, 0] == 0) or np.any(local_coords[:, 0] == sub.shape[0] - 1):
            continue
        if np.any(local_coords[:, 1] == 0) or np.any(local_coords[:, 1] == sub.shape[1] - 1):
            continue
        global_coords = local_coords + np.array([r0, c0], dtype=int)
        border = set()
        for gr, gc in global_coords:
            for dr, dc in DIR4:
                nr, nc = int(gr + dr), int(gc + dc)
                if 0 <= nr < h and 0 <= nc < w and template_mask[nr, nc]:
                    border.add((nr, nc))
        mask = _mask_from_coords(global_coords)
        holes.append(
            {
                "coords": global_coords,
                "mask": mask,
                "key": _canonical_rot_key(mask),
                "border": border,
            }
        )
    return holes


def _extract_pieces(grid: np.ndarray, template: dict, bg: int) -> list[dict]:
    h, w = grid.shape
    keep = np.ones((h, w), dtype=bool)
    keep[template["coords"][:, 0], template["coords"][:, 1]] = False
    remain = (grid != int(bg)) & keep
    pieces = []
    for coords in _components(remain):
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "each loose piece must be monochrome"
        mask = _mask_from_coords(coords)
        pieces.append(
            {
                "coords": coords,
                "mask": mask,
                "key": _canonical_rot_key(mask),
                "color": int(colors[0]),
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered colored pieces to same-shaped template holes under rotation.
    Fill each hole with the piece color and recolor the adjacent grey border ring
    around that hole with the same color. Remove the loose pieces.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    template = _find_template(g, bg)
    holes = _extract_holes(g, template, bg)
    pieces = _extract_pieces(g, template, bg)

    assert holes, "template must contain at least one enclosed hole"
    assert len(holes) == len(pieces), "number of holes must match number of pieces"

    piece_by_key = {}
    for piece in pieces:
        assert piece["key"] not in piece_by_key, "piece shapes must be unique up to rotation"
        piece_by_key[piece["key"]] = piece

    out = g.copy()
    piece_coords = np.concatenate([piece["coords"] for piece in pieces], axis=0)
    out[piece_coords[:, 0], piece_coords[:, 1]] = int(bg)

    for hole in holes:
        piece = piece_by_key.get(hole["key"])
        assert piece is not None, "every hole must have a matching piece"
        color = int(piece["color"])
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = color
        for br, bc in hole["border"]:
            out[br, bc] = color

    return out


_RNG = random.Random()
_GREY = 5
_GEN_ATTEMPTS = 300
_BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        area = rng.randint(3, min(h * w, 7))
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        frontier = set()
        for dr, dc in DIR4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while int(mask.sum()) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for dr, dc in DIR4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if int(mask.sum()) != area:
            continue
        mask = _trim_mask(mask)
        if mask.shape[0] == 1 and mask.shape[1] == 1:
            continue
        rot_keys = {_mask_key(np.rot90(mask, k)) for k in range(4)}
        if len(rot_keys) < 2:
            continue
        if int(mask.sum()) == mask.shape[0] * mask.shape[1] and rng.random() < 0.7:
            continue
        return mask
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for dr, dc in DIR8:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _place_holes(rng: random.Random, template_h: int, template_w: int, shapes: list[np.ndarray]) -> np.ndarray | None:
    hole_mask = np.zeros((template_h, template_w), dtype=bool)
    reserved = np.zeros((template_h, template_w), dtype=bool)
    for shape in sorted(shapes, key=lambda m: int(m.sum()), reverse=True):
        sh, sw = shape.shape
        placed = False
        for _ in range(120):
            if template_h - sh - 1 < 1 or template_w - sw - 1 < 1:
                return None
            r0 = rng.randint(1, template_h - sh - 1)
            c0 = rng.randint(1, template_w - sw - 1)
            cells = np.zeros((template_h, template_w), dtype=bool)
            cells[r0 : r0 + sh, c0 : c0 + sw] = shape
            spread = _dilate8(cells)
            if np.any(reserved & spread):
                continue
            hole_mask |= cells
            reserved |= spread
            placed = True
            break
        if not placed:
            return None
    return hole_mask


def _can_paste(mask: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    ph, pw = patch_mask.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    area = mask[r0 : r0 + ph, c0 : c0 + pw]
    if np.any(area & patch_mask):
        return False
    return True


def _scatter_pieces(
    rng: random.Random,
    grid: np.ndarray,
    split_col: int,
    pieces: list[tuple[np.ndarray, int]],
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, :split_col] = True
    for patch, color in sorted(pieces, key=lambda item: int(item[0].sum()), reverse=True):
        ph, pw = patch.shape
        placed = False
        for _ in range(150):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(split_col, w - pw - 1)
            test = np.zeros((h, w), dtype=bool)
            test[r0 : r0 + ph, c0 : c0 + pw] = patch
            if np.any(_dilate8(test) & occupied):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw][patch] = int(color)
            occupied |= _dilate8(test)
            placed = True
            break
        if not placed:
            return False
    return True


def _largest_component_area(grid: np.ndarray, bg: int) -> int:
    best = 0
    for color in np.unique(grid):
        if int(color) == int(bg):
            continue
        for comp in _components(grid == int(color)):
            best = max(best, int(len(comp)))
    return best


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = int(rng.choice(_BG_CHOICES))
    colors = [c for c in range(10) if c not in {bg, _GREY}]

    piece_count = rng.randint(2, 4)
    shapes = []
    keys = set()
    for _ in range(80):
        if len(shapes) == piece_count:
            break
        shape = _random_shape(rng)
        if shape is None:
            continue
        key = _canonical_rot_key(shape)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(shape)
    if len(shapes) != piece_count:
        return None

    max_sh = max(shape.shape[0] for shape in shapes)
    max_sw = max(shape.shape[1] for shape in shapes)
    template_h = rng.randint(max(9, max_sh + 5), 16)
    template_w = rng.randint(max(10, max_sw + 6), 18)
    hole_mask = _place_holes(rng, template_h, template_w, shapes)
    if hole_mask is None:
        return None

    left_w = template_w + rng.randint(4, 7)
    right_w = max(left_w + rng.randint(-2, 3), max_sw + 8, 12)
    grid_h = max(template_h + rng.randint(4, 8), max_sh + 6, 16)
    grid_w = left_w + right_w
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    template_r0 = rng.randint(1, grid_h - template_h - 1)
    template_c0 = rng.randint(1, left_w - template_w - 1)
    template = np.full((template_h, template_w), _GREY, dtype=int)
    template[hole_mask] = bg
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = template

    piece_colors = rng.sample(colors, piece_count)
    pieces = []
    for shape, color in zip(shapes, piece_colors):
        rot = rng.randrange(4)
        patch_mask = np.rot90(shape, rot)
        pieces.append((patch_mask, color))

    split_col = left_w
    if not _scatter_pieces(rng, grid, split_col, pieces):
        return None

    if _bg_color(grid) != bg:
        return None
    if _largest_component_area(grid, bg) <= max(int(shape.sum()) for shape in shapes):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(_RNG)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
